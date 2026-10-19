"""Configurable Chat Completions transport; credentials stay in the environment."""

import base64
import json
import mimetypes
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def read_config(path):
    """Load public settings; authentication is supplied through the environment."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be an object")

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {
                    "api_key",
                    "token",
                    "authorization",
                    "password",
                    "secret",
                    "headers",
                }:
                    raise ValueError(
                        "Use api_key_env; do not embed credentials or headers"
                    )
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(config)
    if not isinstance(config.get("model"), str) or not config["model"].strip():
        raise ValueError("Specify a model name")
    generation = config.get("generation", {})
    if not isinstance(generation, dict) or set(generation) & {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "stream",
    }:
        raise ValueError("Generation settings cannot override model inputs or tools")
    if not config.get("adapter"):
        endpoint = urlsplit(config.get("base_url", ""))
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.hostname
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError(
                "Specify an HTTP API base without credentials, query or fragment"
            )
    return config


# Settings that change how requests are sent, not what the model is asked.
TRANSPORT_KEYS = ("timeout", "retries", "api_key_env")


def configuration_identity(config):
    return {k: v for k, v in config.items() if k not in TRANSPORT_KEYS}


@contextmanager
def neutral_media(path):
    """Present a video as video.mp4; rendered file names carry the claim condition."""
    path = Path(path)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        target = Path(directory) / ("video" + path.suffix)
        try:
            os.link(path, target)
        except OSError:
            shutil.copyfile(path, target)
        yield target


class Client:
    def __init__(self, config):
        self.config = config
        self.model = config["model"]
        self.key_env = config.get("api_key_env", "MODEL_API_KEY")
        if not os.environ.get(self.key_env):
            raise ValueError(f"Set the API key in the {self.key_env} environment variable")
        self.http = httpx.Client(timeout=float(config.get("timeout", 7200)))

    def close(self):
        self.http.close()

    def publish(self, path):
        """Default transport sends local video bytes, not a machine path."""
        path = Path(path)
        mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode(
            "ascii"
        )

    def __call__(self, messages, *, tools=None, kind=None, **unused):
        token = os.environ[self.key_env]
        body = {
            "model": self.model,
            "messages": messages,
            **self.config.get("generation", {}),
        }
        if tools:
            body.update(tools=tools, tool_choice="required")
        else:
            body.pop("tools", None)
            body.pop("tool_choice", None)
        response = None
        retries = max(0, int(self.config.get("retries", 6)))
        for attempt in range(retries + 1):
            try:
                response = self.http.post(
                    self.config["base_url"].rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + token},
                    json=body,
                )
            except httpx.TransportError:
                if attempt == retries:
                    raise RuntimeError("model_transport_error") from None
                time.sleep(min(60, 2**attempt))
                continue
            if (
                response.status_code not in (429, 500, 502, 503, 504)
                or attempt == retries
            ):
                break
            retry_after = response.headers.get("retry-after", "")
            delay = float(retry_after) if retry_after.isdigit() else min(60, 2**attempt)
            time.sleep(delay)
        if response.status_code != 200:
            raise RuntimeError(f"model_http_{response.status_code}: {response.text[:300]}")
        payload = response.json()
        choice = payload["choices"][0]
        return choice["message"], {
            "finish_reason": choice["finish_reason"],
            "model": payload.get("model"),
            "usage": payload.get("usage"),
            "kind": kind,
        }


def create_client(config):
    if config.get("adapter"):
        import importlib

        module, factory = config["adapter"].split(":", 1)
        return getattr(importlib.import_module(module), factory)(config)
    return Client(config)


def load_client(config_path):
    return create_client(read_config(config_path))
