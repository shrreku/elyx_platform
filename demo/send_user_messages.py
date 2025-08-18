#!/usr/bin/env python3
"""
Send user messages (from a markdown file) to the backend /chat endpoint one-by-one,
waiting for each response before sending the next.

Usage:
    python demo/send_user_messages.py --input user_messages.md --base http://localhost:8000 --out data/sent_messages.jsonl

Behavior:
- Parses a simple markdown format where each message block begins with:
    * Content: "..."
    * sender: "..."
    * date: "YYYY-MM-DD"
    * time: "HH:MM AM/PM"
  (fields other than Content are optional)
- Sends each message to POST {base}/chat and waits for the response.
- Appends a JSON line per message to the output file with:
    {"message": ..., "sender": ..., "date": ..., "time": ..., "request": {...}, "response": {...}}
- Retries transient HTTP errors (3 attempts) with backoff.
- Logs progress to stdout.

Notes:
- Ensure your backend server is running and reachable at --base (default: http://localhost:8000).
- The script is conservative by default (use_crewai=False). You can enable with --use-crewai.
"""

from __future__ import annotations
import argparse
import json
import re
import time
import sys
from typing import Dict, List, Optional
import requests
from pathlib import Path

CONTENT_RE = re.compile(r'^\*\s*Content:\s*"(.*)"\s*$', re.IGNORECASE)
SENDER_RE = re.compile(r'^\*\s*sender:\s*"(.*)"\s*$', re.IGNORECASE)
DATE_RE = re.compile(r'^\*\s*date:\s*"(.*)"\s*$', re.IGNORECASE)
TIME_RE = re.compile(r'^\*\s*time:\s*"(.*)"\s*$', re.IGNORECASE)


def parse_user_messages(md_path: Path) -> List[Dict]:
    """Parse messages from the provided markdown file into a list of dicts.

    Each message dict: {"content": str, "sender": Optional[str], "date": Optional[str], "time": Optional[str]}
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    msgs: List[Dict] = []
    current = None

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        m = CONTENT_RE.match(ln)
        if m:
            # start a new message
            if current:
                msgs.append(current)
            current = {"content": m.group(1).strip(), "sender": None, "date": None, "time": None}
            continue
        if current is None:
            continue
        m = SENDER_RE.match(ln)
        if m:
            current["sender"] = m.group(1).strip()
            continue
        m = DATE_RE.match(ln)
        if m:
            current["date"] = m.group(1).strip()
            continue
        m = TIME_RE.match(ln)
        if m:
            current["time"] = m.group(1).strip()
            continue
        # ignore unknown lines

    if current:
        msgs.append(current)
    return msgs


def send_chat(base_url: str, sender: str, message: str, context: Optional[Dict], use_crewai: bool, max_retries: int = 3, timeout: int = 30) -> Dict:
    """
    Send a single chat request. Special-case 422 responses (validation errors) so they are surfaced
    immediately (not retried), and include response body for debugging.
    """
    url = base_url.rstrip("/") + "/chat"
    # Build payload but omit the 'context' key entirely when it's None to avoid triggering unexpected validation issues
    payload: Dict[str, object] = {"sender": sender, "message": message, "use_crewai": bool(use_crewai)}
    if context is not None:
        payload["context"] = context
    headers = {"Content-Type": "application/json"}
    attempt = 0
    backoffs = [1, 2, 4]
    while attempt < max_retries:
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            # If FastAPI returns 422, it's a validation error - surface it immediately
            if r.status_code == 422:
                try:
                    body = r.json()
                except Exception:
                    body = {"text": r.text}
                raise RuntimeError(f"422 Unprocessable Entity from server: {body}")
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                return {"raw_text": r.text}
        except RuntimeError:
            # Do not retry on 422 validation/runtime-level errors
            raise
        except requests.RequestException as exc:
            attempt += 1
            # For non-validation HTTP errors, retry a few times
            if attempt >= max_retries:
                raise
            wait = backoffs[min(attempt - 1, len(backoffs) - 1)]
            print(f"[WARN] Request failed (attempt {attempt}/{max_retries}): {exc}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to user_messages.md")
    parser.add_argument("--base", "-b", default="http://localhost:8000", help="Backend base URL (default: http://localhost:8000)")
    parser.add_argument("--out", "-o", default="data/sent_messages.jsonl", help="Output JSONL file to append results")
    parser.add_argument("--use-crewai", action="store_true", help="Set use_crewai True in requests")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay (seconds) after each message response before next send (default 0.2s)")
    args = parser.parse_args()

    md_path = Path(args.input)
    if not md_path.exists():
        print(f"Input file not found: {md_path}", file=sys.stderr)
        raise SystemExit(2)

    messages = parse_user_messages(md_path)
    print(f"Parsed {len(messages)} messages from {md_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    for idx, m in enumerate(messages, start=1):
        sender = m.get("sender") or "Rohan"
        content = m.get("content") or ""
        context = {"date": m.get("date"), "time": m.get("time")} if (m.get("date") or m.get("time")) else None

        record = {
            "index": idx,
            "message": content,
            "sender": sender,
            "context": context,
            "request": None,
            "response": None,
            "error": None,
            "timestamp": time.time()
        }

        print(f"[{idx}/{len(messages)}] Sending message from {sender}: {content[:80]}{'...' if len(content)>80 else ''}")
        try:
            res = send_chat(args.base, sender, content, context, args.use_crewai)
            record["request"] = {"sender": sender, "message": content, "context": context, "use_crewai": args.use_crewai}
            record["response"] = res
            # append to file
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            record["error"] = str(exc)
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[ERROR] Failed to send message {idx}: {exc}", file=sys.stderr)
            # continue to next message, but do not flood
        # Ensure we wait until response is processed; small delay to be gentle with server
        time.sleep(args.delay)

    print("Done sending messages. Records saved to", out_path)


if __name__ == "__main__":
    main()
