import os
import json
from typing import List, Dict, Optional
import jsonschema
import requests
from dotenv import load_dotenv


load_dotenv()


class BaseAgent:
    def __init__(self, name: str, role: str, system_prompt: str, schema: Optional[Dict] = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.schema = schema
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
                return body["choices"][0]["message"]["content"]
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

        for i in range(3):  # Retry up to 3 times
            response_text = self.call_openrouter(messages)
            try:
                # Attempt to extract a valid JSON object from the response
                json_part = response_text[response_text.find('{'):response_text.rfind('}')+1]
                response_json = json.loads(json_part)

                if self.schema:
                    jsonschema.validate(instance=response_json, schema=self.schema)

                self.conversation_history.append({"user": user_message, "assistant": json.dumps(response_json)})
                return json.dumps(response_json)
            except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                error_message = f"Attempt {i+1} failed: Invalid JSON or schema validation error: {e}\nResponse: {response_text}"
                messages.append({"role": "system", "content": error_message})
                continue

        # If all retries fail, return an error message
        final_error = "Failed to get a valid JSON response after multiple retries."
        self.conversation_history.append({"user": user_message, "assistant": final_error})
        return final_error

