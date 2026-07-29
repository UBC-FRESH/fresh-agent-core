"""Error hierarchy for fresh-agent-core."""

from __future__ import annotations


class FreshAgentError(Exception):
    """Base class for every error raised by this package."""


class AgentUnavailable(FreshAgentError):
    """
    No usable agent configuration was found.

    Raised when a capability is invoked but no endpoint/model has been configured.
    Callers that want to branch rather than catch should use
    :py:func:`fresh_agent_core.available` instead.
    """


class ProviderError(FreshAgentError):
    """
    The model provider could not be reached, or returned an unusable response.

    Covers transport failures, non-2xx responses, and malformed response payloads.
    Deliberately distinct from :py:class:`ValidationExhausted`: this means the
    *provider* failed, not that the model produced invalid content.
    """


class ValidationExhausted(FreshAgentError):
    """
    Every attempt produced output that failed validation.

    Raised only by callers that opt into exceptions; the default path returns a
    ``CapabilityResult`` with ``ok=False`` so that failure is data rather than
    control flow.
    """
