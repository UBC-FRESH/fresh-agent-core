"""
Model provider clients.

A provider does one thing: turn a list of chat messages into a completion string.
It performs no validation and makes no judgement about content -- that is the
capability's job, and keeping the split clean is what lets the whole test suite run
offline against :py:class:`~fresh_agent_core.testing.FakeProvider`.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Protocol, runtime_checkable

from fresh_agent_core.config import AgentConfig
from fresh_agent_core.errors import ProviderError

#: Transport-level retries. Deliberately small and only for transient failures --
#: this is not the capability retry budget, which is a separate concept and lives
#: in the capability loop.
DEFAULT_TRANSPORT_RETRIES = 2

#: HTTP statuses worth retrying: rate limiting and transient server faults.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@runtime_checkable
class Provider(Protocol):
    """Minimal contract a model backend must satisfy."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        Return a completion for *messages*.

        :param messages: OpenAI-style chat messages.
        :return: The assistant's message content.
        :raises ProviderError: On transport failure or an unusable response.
        """
        ...


class OpenAIProvider:
    """
    Client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Covers vLLM, llama.cpp servers, Ollama's OpenAI shim, and OpenAI itself.

    :param config: Endpoint, model, credentials and limits.
    :param transport_retries: Retries for *transient* transport failures only.
    """

    def __init__(
        self,
        config: AgentConfig,
        transport_retries: int = DEFAULT_TRANSPORT_RETRIES,
    ) -> None:
        self.config = config
        self.transport_retries = transport_retries

    def _url(self) -> str:
        return f"{self.config.endpoint.rstrip('/')}/chat/completions"

    def _payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            'model': self.config.model,
            'messages': messages,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
        }

    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        Return a completion, retrying only transient transport failures.

        Errors are raised as :py:class:`ProviderError` with the endpoint *host*
        (never the full URL, which can carry credentials) and never the request
        headers.
        """
        import httpx  # imported lazily so `import fresh_agent_core` stays cheap

        last_error: Optional[str] = None
        attempts = self.transport_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(
                    self._url(),
                    json=self._payload(messages),
                    headers=self.config.request_headers(),
                    timeout=self.config.timeout,
                )
            except httpx.HTTPError as exc:
                last_error = f'{type(exc).__name__}: {exc}'
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise ProviderError(
                    f'Could not reach {self.config.endpoint_host} after {attempts} '
                    f'attempt(s): {last_error}'
                ) from exc

            if response.status_code in _RETRYABLE_STATUS and attempt < attempts:
                last_error = f'HTTP {response.status_code}'
                time.sleep(min(2 ** (attempt - 1), 8))
                continue

            if response.status_code >= 400:
                raise ProviderError(
                    f'{self.config.endpoint_host} returned HTTP '
                    f'{response.status_code} for model {self.config.model!r}.'
                )

            return _extract_content(response.json(), self.config.endpoint_host)

        # Only reachable if every attempt was retryable and the budget ran out.
        raise ProviderError(
            f'{self.config.endpoint_host} did not return a usable response after '
            f'{attempts} attempt(s). Last failure: {last_error}'
        )


def _extract_content(body: Any, host: str) -> str:
    """
    Pull the assistant message out of an OpenAI-shaped response body.

    Validated structurally rather than trusted: a malformed body is a provider
    problem and should say so, instead of surfacing later as a confusing
    ``TypeError`` or ``KeyError`` from inside a capability's parser.
    """
    if not isinstance(body, dict):
        raise ProviderError(f'{host} returned {type(body).__name__}, expected a JSON object.')

    choices = body.get('choices')
    if not isinstance(choices, list) or not choices:
        raise ProviderError(f"{host} returned a response with no 'choices'.")

    message = choices[0].get('message') if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ProviderError(f"{host} returned a choice with no 'message' object.")

    content = message.get('content')
    if not isinstance(content, str):
        raise ProviderError(
            f"{host} returned a message whose 'content' is "
            f'{type(content).__name__}, expected a string.'
        )
    return content
