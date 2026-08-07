"""
Configuration resolution for fresh-agent-core.

Resolution order, first hit wins:

1. an explicit :py:class:`AgentConfig` passed by the caller
2. environment variables
3. a user config file at ``~/.config/fresh-agent/config.toml``
4. otherwise **unavailable** -- :py:func:`available` returns ``False`` and
   capabilities raise :py:class:`~fresh_agent_core.errors.AgentUnavailable`

Nothing about any particular endpoint is hardcoded. Credentials are read from the
environment or the user config file only, never from a repository.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# tomllib landed in 3.11. ws3 supports 3.10, so its optional extras must too --
# raising this package's floor would make the agent extra unusable for part of
# ws3's supported range.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

ENV_ENDPOINT = 'FRESH_AGENT_ENDPOINT'
ENV_MODEL = 'FRESH_AGENT_MODEL'
ENV_API_KEY = 'FRESH_AGENT_API_KEY'
ENV_HEADERS = 'FRESH_AGENT_HEADERS'
ENV_TIMEOUT = 'FRESH_AGENT_TIMEOUT'

DEFAULT_CONFIG_PATH = Path.home() / '.config' / 'fresh-agent' / 'config.toml'
DEFAULT_TIMEOUT = 60.0

#: Header names whose values are redacted in reprs and provenance records.
#: Matching is case-insensitive and by substring, so vendor-specific names such as
#: ``CF-Access-Client-Secret`` are covered without enumerating them.
_SECRET_HINTS = ('secret', 'token', 'key', 'auth', 'password', 'cookie')

_REDACTED = '<redacted>'


def _is_secret(name: str) -> bool:
    """True if a header name looks like it carries credentials."""
    lowered = name.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Return a copy of *headers* with credential-bearing values replaced.

    Used for reprs and provenance. Applied by substring match on the header name so
    that unfamiliar vendor headers are redacted by default rather than leaked by
    omission.
    """
    return {k: (_REDACTED if _is_secret(k) else v) for k, v in headers.items()}


@dataclass(frozen=True)
class AgentConfig:
    """
    Everything needed to reach a model endpoint.

    :param endpoint: Base URL of an OpenAI-compatible API, e.g.
        ``https://host/v1``.
    :param model: Model identifier as the endpoint expects it.
    :param api_key: Optional bearer token.
    :param headers: Extra headers, e.g. for an access proxy.
    :param timeout: Per-request timeout in seconds.
    :param temperature: Sampling temperature. Defaults to 0 -- capabilities want
        reproducible proposals, not creative ones.
    :param max_tokens: Upper bound on completion length.
    """

    endpoint: str
    model: str
    api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT
    temperature: float = 0.0
    max_tokens: int = 4096

    @property
    def endpoint_host(self) -> str:
        """
        Host portion of the endpoint.

        Provenance records this rather than the full URL, since a URL can carry
        credentials in userinfo or query parameters.
        """
        return urlsplit(self.endpoint).netloc or self.endpoint

    def request_headers(self) -> dict[str, str]:
        """Headers for an outgoing request, including auth. Never log this."""
        out = dict(self.headers)
        if self.api_key:
            out['Authorization'] = f'Bearer {self.api_key}'
        return out

    def safe_headers(self) -> dict[str, str]:
        """Headers with credential values redacted. Safe to log."""
        return redact_headers(self.request_headers())

    def __repr__(self) -> str:
        """
        Redacted repr.

        Defined explicitly because the default dataclass repr would print the API
        key and every header value, and configs end up in tracebacks and logs.
        """
        return (
            f'AgentConfig(endpoint={self.endpoint!r}, model={self.model!r}, '
            f'api_key={_REDACTED if self.api_key else None!r}, '
            f'headers={self.safe_headers()!r}, timeout={self.timeout!r}, '
            f'temperature={self.temperature!r}, max_tokens={self.max_tokens!r})'
        )

    __str__ = __repr__


def _from_env() -> AgentConfig | None:
    """Build a config from environment variables, or None if incomplete."""
    endpoint = os.environ.get(ENV_ENDPOINT)
    model = os.environ.get(ENV_MODEL)
    if not endpoint or not model:
        return None

    headers: dict[str, str] = {}
    raw_headers = os.environ.get(ENV_HEADERS)
    if raw_headers:
        try:
            parsed = json.loads(raw_headers)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f'{ENV_HEADERS} must be a JSON object mapping header names to '
                f'values; could not parse it: {exc}'
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(f'{ENV_HEADERS} must be a JSON object, got {type(parsed).__name__}')
        headers = {str(k): str(v) for k, v in parsed.items()}

    timeout = DEFAULT_TIMEOUT
    raw_timeout = os.environ.get(ENV_TIMEOUT)
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ValueError(f'{ENV_TIMEOUT} must be a number, got {raw_timeout!r}') from exc

    return AgentConfig(
        endpoint=endpoint,
        model=model,
        api_key=os.environ.get(ENV_API_KEY),
        headers=headers,
        timeout=timeout,
    )


def _from_file(path: Path) -> AgentConfig | None:
    """Build a config from a TOML file, or None if absent or incomplete."""
    if not path.is_file():
        return None
    with path.open('rb') as handle:
        data: dict[str, Any] = tomllib.load(handle)

    section = data.get('agent', data)
    endpoint = section.get('endpoint')
    model = section.get('model')
    if not endpoint or not model:
        return None

    raw_headers = section.get('headers', {}) or {}
    return AgentConfig(
        endpoint=str(endpoint),
        model=str(model),
        api_key=section.get('api_key'),
        headers={str(k): str(v) for k, v in raw_headers.items()},
        timeout=float(section.get('timeout', DEFAULT_TIMEOUT)),
        temperature=float(section.get('temperature', 0.0)),
        max_tokens=int(section.get('max_tokens', 4096)),
    )


def resolve(
    config: AgentConfig | None = None,
    *,
    config_path: Path | None = None,
) -> AgentConfig | None:
    """
    Resolve a configuration, or return ``None`` if none is available.

    Returns ``None`` rather than raising so that callers can degrade gracefully.
    Use :py:func:`available` for a boolean probe.

    :param config: An explicit config, which short-circuits resolution.
    :param config_path: Override the user config file location, mainly for tests.
    """
    if config is not None:
        return config
    from_env = _from_env()
    if from_env is not None:
        return from_env
    return _from_file(config_path or DEFAULT_CONFIG_PATH)


def available(
    config: AgentConfig | None = None,
    *,
    config_path: Path | None = None,
) -> bool:
    """
    True when a configuration can be resolved.

    Deliberately never raises and never touches the network: it answers "is this
    configured", not "is the endpoint reachable". Reachability is only knowable by
    making a call, and this probe must be cheap enough to sit in an ``if``.
    """
    try:
        return resolve(config, config_path=config_path) is not None
    except (ValueError, OSError, tomllib.TOMLDecodeError):
        # Malformed configuration is treated as absent. Raising here would make a
        # cheap probe throw from inside an `if`, which is exactly what callers are
        # using it to avoid.
        return False
