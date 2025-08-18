from __future__ import annotations

import json
from typing import Dict, List, Optional

from .base_agent import BaseAgent


class IssueExtractor:
    """LLM-backed extractor to identify problems/issues and improvements from a user message.

    This class provides two APIs:
    - extract_all(message, context) -> {"issues": [...], "improvements": [...]}
    - extract(message, context) -> List[issues]  (backwards-compatible; calls extract_all)
    """

    def __init__(self):
        self.agent = BaseAgent(
            name="IssueExtractor",
            role="Classifier",
            system_prompt=(
                """
You are a concise classifier that extracts PROBLEMS/ISSUES/IMPROVEMENTS from a user's message.
Output STRICT JSON with shape:
{
  "issues": [
    {"title": str, "details": str, "category": str, "severity": "low|medium|high"}
  ],
  "improvements": [
    {"title": str, "details": str, "related_issue_title": Optional[str]}
  ]
}
Rules:
- If none found, return empty lists: {"issues": [], "improvements": []}.
- categories to prefer for issues: "medical", "nutrition", "physio", "logistics", "performance", "other".
- Keep titles short (<=80 chars). Details <= 240 chars.
- For improvements, include related_issue_title if the improvement obviously maps to a previously reported issue (optional).
"""
            ),
        )

    def build_messages(self, message: str, context: Optional[Dict] = None) -> List[Dict[str, str]]:
        u = {
            "role": "user",
            "content": (
                f"User message: {message}\n\n"
                f"Context: {json.dumps(context) if context else '{}'}\n\n"
                "Extract issues and improvements now in STRICT JSON as described."
            ),
        }
        return [
            {"role": "system", "content": self.agent.system_prompt},
            u,
        ]

    def _parse_json(self, raw: str) -> Dict:
        try:
            return json.loads(raw)
        except Exception:
            import re

            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                return {}
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}

    def extract_all(self, message: str, context: Optional[Dict] = None) -> Dict[str, List[Dict]]:
        """Return both issues and improvements as a dict: {'issues': [...], 'improvements': [...] }"""
        msgs = self.build_messages(message, context)
        raw = self.agent.call_openrouter(msgs)
        data = self._parse_json(raw)

        issues = data.get("issues") if isinstance(data.get("issues"), list) else []
        improvements = data.get("improvements") if isinstance(data.get("improvements"), list) else []

        cleaned_issues: List[Dict] = []
        for it in issues:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title", "")).strip()
            details = str(it.get("details", "")).strip()
            category = str(it.get("category", "other")).strip() or "other"
            severity = str(it.get("severity", "medium")).strip().lower()
            if not title:
                continue
            if severity not in {"low", "medium", "high"}:
                severity = "medium"
            cleaned_issues.append(
                {
                    "title": title[:200],
                    "details": details[:500],
                    "category": category[:40],
                    "severity": severity,
                }
            )

        cleaned_improvements: List[Dict] = []
        for it in improvements:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title", "")).strip()
            details = str(it.get("details", "")).strip()
            related = str(it.get("related_issue_title", "")).strip() or None
            if not title:
                continue
            cleaned_improvements.append(
                {
                    "title": title[:200],
                    "details": details[:500],
                    "related_issue_title": related,
                }
            )

        return {"issues": cleaned_issues, "improvements": cleaned_improvements}

    def extract(self, message: str, context: Optional[Dict] = None) -> List[Dict]:
        """Backwards-compatible: return only 'issues' as a list."""
        res = self.extract_all(message, context)
        return res.get("issues", [])
