import os
import json
from typing import List, Dict

import requests
from dotenv import load_dotenv


load_dotenv()


class BaseAgent:
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict] = []

    def _should_use_mock(self) -> bool:
        use_mock = os.getenv("USE_MOCK_RESPONSES", "0").strip() in {"1", "true", "True"}
        return use_mock or not os.getenv("OPENROUTER_API_KEY")

    def _mock_response(self, messages: List[Dict]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[{self.name} - {self.role}] Acknowledged: {last_user[:160]}"

    def call_openrouter(self, messages: List[Dict], model: str | None = None) -> str:  # type: ignore[valid-type]
        if self._should_use_mock():
            return self._mock_response(messages)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            # Optional but recommended by OpenRouter
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Elyx Simulation"),
        }

        selected_model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
        data = {
            "model": selected_model,
            "messages": messages,
            "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0.7")),
        }

        url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

        # Simple retry/backoff for 429s
        backoffs = [1, 2, 4]
        for attempt, delay in enumerate([0] + backoffs):
            if delay:
                try:
                    import time
                    time.sleep(delay)
                except Exception:
                    pass
            response = requests.post(url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                body = response.json()
                # Handle different response formats
                if "choices" in body and len(body["choices"]) > 0:
                    return body["choices"][0]["message"]["content"]
                elif "content" in body:
                    return body["content"]
                elif "response" in body:
                    return body["response"]
                else:
                    # Log the unexpected response format for debugging
                    print(f"Unexpected API response format: {json.dumps(body, indent=2)}")
                    raise RuntimeError(f"Unexpected API response format - missing 'choices' field")
            # On 429 keep retrying; otherwise break
            if response.status_code != 429:
                break

        # No mock fallback; propagate final error
        raise RuntimeError(f"OpenRouter API Error: {response.status_code} - {response.text[:200]}")

    def respond(self, user_message: str, context: Dict | None = None) -> str:  # type: ignore[valid-type]
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        if context:
            messages.insert(1, {"role": "system", "content": f"Context: {json.dumps(context)}"})

        response = self.call_openrouter(messages)
        self.conversation_history.append({"user": user_message, "assistant": response})
        return response
