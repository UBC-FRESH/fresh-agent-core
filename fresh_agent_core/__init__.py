"""
Shared runtime for embedded agent capabilities across the UBC-FRESH ecosystem.

The premise, stated once here because everything else follows from it:

    A capability is a prompt plus a validator plus a retry budget.
    No oracle, no capability.

An LLM is used as a *proposal generator* inside a closed loop with a hard oracle::

    build prompt -> call model -> parse -> validate against real state
                        ^                          |
                        +------ feed failure back --+   (bounded retries)

Output that fails validation never reaches the caller. On exhaustion the capability
returns ``ok=False`` with the accumulated errors, never a best guess.

This package owns the *mechanism*: configuration, the provider client, the
``Capability`` contract and its retry loop, provenance, a test double, and an MCP
host. It deliberately owns no domain knowledge -- the validator is the
domain-specific part, and lives in the adopting package. Only ws3 knows what makes
a ws3 mask valid.

``fresh_agent_core`` must never import ws3, femic, fhops, or freshforge.
"""

__version__ = '0.1.0a1'

__all__ = [
    'AgentConfig',
    'AgentUnavailable',
    'Capability',
    'CapabilityResult',
    'FakeProvider',
    'FreshAgentError',
    'JSONLSink',
    'MemorySink',
    'NullSink',
    'OpenAIProvider',
    'ParseError',
    'ProvenanceRecord',
    'ProvenanceSink',
    'Provider',
    'ProviderError',
    'Registry',
    'ValidationExhausted',
    'Verdict',
    'available',
]

from .capability import Capability, CapabilityResult, ParseError, Verdict
from .config import AgentConfig, available
from .errors import (
    AgentUnavailable,
    FreshAgentError,
    ProviderError,
    ValidationExhausted,
)
from .provenance import JSONLSink, MemorySink, NullSink, ProvenanceRecord, ProvenanceSink
from .provider import OpenAIProvider, Provider
from .registry import Registry
from .testing import FakeProvider
