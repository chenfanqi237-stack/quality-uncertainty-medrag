"""Small Ollama text backend, currently used only for clinical query reformulation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class OllamaBackendError(RuntimeError):
    """Ollama could not return a complete, usable text response."""


class OllamaTextGenerationBackend:
    """Implement TextGenerationBackend without an SDK or API key.

    Constructor settings select the service, model, timeout, seed and thinking mode.
    Per-call generation settings are copied into Ollama's ``options`` object.
    Calls are independent: no conversation history or model context is reused.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        timeout: float = 180.0,
        seed: int = 42,
        think: bool = False,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a nonempty HTTP(S) URL")
        normalized_url = base_url.strip().rstrip("/")
        try:
            parsed = urlsplit(normalized_url)
            _ = parsed.port  # Validate a supplied port before constructing requests.
        except ValueError as exc:
            raise ValueError("base_url must be a valid HTTP(S) URL") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or any(character.isspace() for character in normalized_url)
        ):
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(think, bool):
            raise ValueError("think must be a boolean")
        self.base_url = normalized_url
        self.model = model.strip()
        self.timeout = float(timeout)
        self.seed = seed
        self.think = think

    @property
    def backend_id(self) -> str:
        return f"ollama/{self.model}"

    def generate(
        self, prompt: str, *, generation_config: Mapping[str, object] | None = None,
    ) -> str:
        """Return only completed final text; query validation belongs to the reformulator."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if generation_config is not None and not isinstance(generation_config, Mapping):
            raise ValueError("generation_config must be a mapping or None")
        options: dict[str, object] = {"temperature": 0.0, "seed": self.seed}
        if generation_config is not None:
            if any(not isinstance(key, str) or not key.strip() for key in generation_config):
                raise ValueError("generation option names must be nonempty strings")
            options.update(generation_config)
        temperature = options["temperature"]
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be a finite nonnegative number")
        if isinstance(options["seed"], bool) or not isinstance(options["seed"], int):
            raise ValueError("seed must be an integer")
        payload = {
            "model": self.model, "prompt": prompt,
            "stream": False, "think": self.think, "options": options,
        }
        try:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("generation options must be finite JSON-serializable values") from exc
        request = Request(
            f"{self.base_url}/api/generate", data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            raise OllamaBackendError(f"Ollama generation failed with HTTP status {exc.code}") from exc
        except (URLError, OSError) as exc:
            raise OllamaBackendError("Ollama generation failed: connection error or timeout") from exc
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise OllamaBackendError("Ollama returned invalid JSON") from exc
        if not isinstance(result, dict) or "error" in result:
            raise OllamaBackendError("Ollama returned an invalid or error response")
        if result.get("done") is not True:
            raise OllamaBackendError("Ollama returned incomplete generation")
        text = result.get("response")
        if not isinstance(text, str) or not text.strip():
            raise OllamaBackendError("Ollama returned missing or empty generated text")
        return text
