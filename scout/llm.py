"""Optional LLM access.

The agent is designed so the model is an *enhancement*, never a requirement:

* retrieval, claim splitting and citation verification are algorithms;
* question decomposition and prose synthesis use the model when a key exists,
  and fall back to deterministic rules when it does not.

That inversion is deliberate. It means the expensive, non-deterministic part
cannot be the only thing standing between the user and a working tool, and it
means every test runs offline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

KNOWN_PROVIDERS = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "env": ["SILICONFLOW_CN_API_KEY", "SILICONFLOW_API_KEY"],
        "default_model": "deepseek-ai/DeepSeek-V3.2",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env": ["DEEPSEEK_API_KEY"],
        "default_model": "deepseek-chat",
    },
    "zenmux": {
        "base_url": "https://zenmux.ai/api/v1",
        "env": ["ZENMUX_API_KEY"],
        "default_model": "deepseek/deepseek-chat",
    },
}

# Keys another local tool already stored, so the user does not paste secrets twice.
FOREIGN_CREDENTIALS = [
    Path.home() / ".local/share/opencode/auth.json",
    Path.home() / ".config/opencode/auth.json",
]

_ALIASES = {"siliconflow-cn": "siliconflow", "moonshotai": "moonshot"}


def _dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    return values


def _foreign_keys() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in FOREIGN_CREDENTIALS:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            for name, entry in data.items():
                if isinstance(entry, dict) and entry.get("key"):
                    found.setdefault(_ALIASES.get(name, name), str(entry["key"]))
    return found


@dataclass
class TraceResult:
    """Shape-compatible with Lens's ChatResult.

    Scout never imports Lens. The tracer only needs `record_llm(result, provider)`
    to read these attributes, so the coupling is structural — Scout still installs
    and runs on its own, and any compatible tracer works.
    """

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


@dataclass
class LLM:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    provider_name: str = ""
    # Anything with a `span(name, kind=...)` context manager. Lens satisfies this
    # duck-typed contract, so Scout can be traced without importing it — the
    # dependency stays optional and Scout still runs standalone.
    tracer: object | None = None

    @classmethod
    def from_env(cls, provider: str | None = None) -> "LLM":
        dotenv = _dotenv()
        name = (provider or os.environ.get("SCOUT_PROVIDER") or dotenv.get("SCOUT_PROVIDER") or "siliconflow").lower()
        name = _ALIASES.get(name, name)
        spec = KNOWN_PROVIDERS.get(name)
        if spec is None:
            return cls()

        key = ""
        for env_name in spec["env"]:
            key = os.environ.get(env_name) or dotenv.get(env_name, "")
            if key:
                break
        if not key:
            key = _foreign_keys().get(name, "")

        model = os.environ.get("SCOUT_MODEL") or dotenv.get("SCOUT_MODEL") or spec["default_model"]
        return cls(base_url=spec["base_url"], api_key=key, model=model, provider_name=name)

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 800) -> str | None:
        """Return None instead of raising when unavailable.

        Callers are written to degrade, so a missing key must look like an absent
        capability rather than an outage.
        """
        if not self.available:
            return None

        if self.tracer is None:
            text, _ = self._request(prompt, system, temperature, max_tokens)
            return text

        # Name the span after the model, not after the caller: on a trace the
        # useful grouping is "which model cost me what", not "which function asked".
        with self.tracer.span(f"llm:{self.model.split('/')[-1]}", kind="llm",
                              input=prompt[:200]) as span:
            text, usage = self._request(prompt, system, temperature, max_tokens)
            if text is None:
                span.set(failed=True)
                return None
            span.record_llm(
                TraceResult(text=text, model=self.model,
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0)),
                self.provider_name or "unknown",
            )
            return text

    def _request(self, prompt: str, system: str | None, temperature: float,
                 max_tokens: int) -> tuple[str | None, dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Sandboxes inject a loopback proxy for their own traffic; opt out unless
        # the operator pins one explicitly.
        trust_env = os.environ.get("SCOUT_TRUST_ENV", "1") != "0"
        kwargs = {"trust_env": trust_env}
        if os.environ.get("SCOUT_PROXY"):
            kwargs["proxy"] = os.environ["SCOUT_PROXY"]

        try:
            with httpx.Client(timeout=120.0, **kwargs) as client:
                resp = client.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model, "messages": messages,
                          "temperature": temperature, "max_tokens": max_tokens},
                )
            if resp.status_code >= 400:
                return None, {}
            data = resp.json()
            choices = data.get("choices") or []
            text = ((choices[0].get("message") or {}).get("content") or "").strip() or None
            return text, data.get("usage") or {}
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError):
            return None, {}

    def complete_json(self, prompt: str, **kwargs) -> dict | list | None:
        """Ask for JSON and survive the ways models wrap it."""
        raw = self.complete(prompt, **kwargs)
        if not raw:
            return None
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
