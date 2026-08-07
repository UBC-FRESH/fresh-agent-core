"""
The capability contract.

    A capability is a prompt plus a validator plus a retry budget.
    No oracle, no capability.

An LLM is used as a *proposal generator* inside a closed loop with a hard oracle::

    build prompt -> call model -> parse -> validate against real state
                        ^                          |
                        +------ feed failure back --+   (bounded retries)

Two properties follow, and they are the whole point:

1. **Unvalidated output never reaches the caller.** On exhaustion the capability
   returns ``ok=False`` with the accumulated errors, never a best guess.
2. **A small model suffices.** With a narrow task, a hard oracle and bounded
   retries, the model only needs to emit plausible candidates cheaply.

Implementers supply three methods. :py:meth:`Capability.validate` is the one that
matters -- it must check the candidate against **real state**, not against a mock
or a regex over the model's own output. A validator that cannot fail is not a
validator, and a capability without a real oracle should not exist.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

from fresh_agent_core.config import AgentConfig
from fresh_agent_core.errors import ValidationExhausted
from fresh_agent_core.provenance import (
    NullSink,
    ProvenanceRecord,
    ProvenanceSink,
    prompt_digest,
)
from fresh_agent_core.provider import Provider

#: Candidate type produced by :py:meth:`Capability.parse`.
T = TypeVar('T')

DEFAULT_MAX_ATTEMPTS = 3


class ParseError(ValueError):
    """
    Raised by :py:meth:`Capability.parse` when a completion is unusable.

    Treated as a validation failure rather than an error: a model that returned
    malformed output may well return usable output next time, and the parse
    message is fed back into the retry prompt.
    """


@dataclass(frozen=True)
class Verdict:
    """
    The outcome of validating one candidate.

    :param ok: Whether the candidate passed.
    :param errors: Why it failed. These are shown *to the model* on the next
        attempt, so they should read as actionable corrections rather than
        internal diagnostics.
    """

    ok: bool
    errors: tuple[str, ...] = ()

    @classmethod
    def valid(cls) -> Verdict:
        """A passing verdict."""
        return cls(True)

    @classmethod
    def invalid(cls, *errors: str) -> Verdict:
        """
        A failing verdict.

        :param errors: At least one reason. A failing verdict with no reason gives
            the retry nothing to work with, so it is rejected.
        """
        if not errors:
            raise ValueError(
                'Verdict.invalid() requires at least one reason. The reasons are fed '
                'back to the model on the next attempt; without them the retry is '
                'just a re-roll.'
            )
        return cls(False, tuple(errors))


@dataclass(frozen=True)
class CapabilityResult(Generic[T]):
    """
    What a capability run produced.

    Failure is data, not an exception: a model that could not produce valid output
    is an expected outcome, and callers should branch on it rather than catch.

    :param ok: Whether a validated result was obtained.
    :param value: The validated candidate, or ``None`` when ``ok`` is False.
    :param attempts: How many model calls were made.
    :param provenance_ids: Record ids for every attempt, in order.
    :param errors: Accumulated validation failures across all attempts.
    """

    ok: bool
    value: T | None
    attempts: int
    provenance_ids: tuple[str, ...]
    errors: tuple[str, ...]

    def unwrap(self) -> T:
        """
        Return the value, or raise if there is none.

        For callers who prefer exceptions at the boundary. The default path
        returns this object so failure stays inspectable.
        """
        if not self.ok or self.value is None:
            raise ValidationExhausted(
                f'No valid result after {self.attempts} attempt(s). '
                f'Failures: ' + '; '.join(self.errors)
            )
        return self.value


class Capability(ABC, Generic[T]):
    """
    Base class for a validated, agent-backed operation.

    :cvar name: Stable identifier, used in provenance and as the MCP tool name.
    :cvar description: What an external agent reads when deciding whether to call
        this. State what the capability *validates*, so the caller knows what
        guarantee it is getting.
    :cvar max_attempts: Retry budget. Distinct from the provider's transport
        retries, which handle a flaky connection rather than unusable content.
    """

    name: str = ''
    description: str = ''
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    #: JSON Schema for this capability's inputs, used to advertise the tool over
    #: MCP. The default accepts any object; override it so a calling agent can see
    #: what the capability actually expects rather than guessing.
    input_schema: ClassVar[dict[str, Any]] = {'type': 'object'}

    def from_payload(self, payload: dict[str, Any]) -> Any:
        """
        Convert a JSON tool-call payload into this capability's input type.

        Defaults to passing the dict through. Override when the capability takes a
        structured input, so that the MCP boundary is the only place that has to
        know about JSON.

        :param payload: Decoded arguments from a tool call.
        """
        return payload

    def coerce_input(self, inputs: Any) -> Any:
        """
        Normalise caller-supplied input into this capability's input type.

        Applied at the top of :py:meth:`run`, so every entry point benefits --
        direct Python calls, MCP tool calls, and tests alike.

        Defaults to passing through. Override to accept convenient shorthand
        alongside the structured type: a bare string where the capability takes a
        single description, or a dict matching the JSON schema.

        Exists because a documented convenience form that the code does not
        actually accept is a documentation defect waiting to happen. Coercion is
        declared here, next to the type it produces, rather than left to each
        caller to remember.
        """
        return inputs

    def render(self, value: T) -> str:
        """
        Render a validated result as text for a tool response.

        Defaults to ``str``. Override to produce something an agent can act on
        directly rather than having to parse a repr.
        """
        return str(value)

    def __init__(self) -> None:
        """
        Reject concrete capabilities that omit identity.

        Checked here rather than in ``__init_subclass__`` because ``ABCMeta`` has
        not yet populated ``__abstractmethods__`` at subclass-creation time. This
        placement also allows intermediate abstract subclasses to omit a name --
        only something you can actually instantiate needs one.
        """
        if not self.name:
            raise TypeError(
                f'{type(self).__name__} must define a non-empty `name`; provenance '
                f'records and MCP tool registration are keyed on it.'
            )

    @abstractmethod
    def build_messages(
        self,
        inputs: Any,
        failures: tuple[str, ...],
    ) -> list[dict[str, str]]:
        """
        Build the chat messages for one attempt.

        :param inputs: Whatever this capability takes.
        :param failures: Validation errors from previous attempts, oldest first.
            Empty on the first attempt. **Incorporate these** -- a retry that
            re-sends the identical prompt is just a re-roll and wastes the budget.
        """

    @abstractmethod
    def parse(self, raw: str) -> T:
        """
        Turn a raw completion into a candidate.

        :raises ParseError: If the completion is unusable. The message is fed back
            into the next attempt, so make it specific about what was expected.
        """

    @abstractmethod
    def validate(self, candidate: T, context: Any) -> Verdict:
        """
        Check *candidate* against real state.

        This is the oracle, and the reason the whole approach works. It must
        consult actual state -- resolve the mask against the model, re-parse the
        file, confirm the symbol exists. Validating the model's output against
        another model, a regex over its own text, or a mock proves nothing.

        :param candidate: Parsed output from :py:meth:`parse`.
        :param context: Domain state to validate against, passed through from
            :py:meth:`run`.
        """

    def run(
        self,
        inputs: Any,
        *,
        provider: Provider,
        config: AgentConfig,
        context: Any = None,
        sink: ProvenanceSink | None = None,
    ) -> CapabilityResult[T]:
        """
        Execute the validate/retry loop.

        Provider failures are *not* caught: a transport problem is an
        infrastructure fault the caller cannot fix by retrying content, and it
        propagates as :py:class:`~fresh_agent_core.errors.ProviderError`. Content
        failures -- unparseable or invalid output -- are absorbed, fed back, and
        ultimately reported as ``ok=False``.

        :param inputs: Passed to :py:meth:`build_messages`.
        :param provider: Model backend.
        :param config: Used for provenance metadata only.
        :param context: Passed to :py:meth:`validate`.
        :param sink: Where provenance goes. Defaults to discarding.
        :return: A result that is either validated or explicitly unsuccessful.
        """
        # `is None`, not `or`: sinks may define __len__, and an empty one is then
        # falsy, so `sink or NullSink()` would silently discard the caller's sink
        # on the very first call -- when it is empty by definition.
        if sink is None:
            sink = NullSink()
        inputs = self.coerce_input(inputs)
        failures: list[str] = []
        provenance_ids: list[str] = []

        for attempt in range(1, self.max_attempts + 1):
            messages = self.build_messages(inputs, tuple(failures))

            started = time.monotonic()
            raw = provider.complete(messages)
            duration_ms = (time.monotonic() - started) * 1000.0

            # `parsed` is bound only on the success path, so it stays typed as T
            # and can be passed to validate() without a cast. `candidate` carries
            # the optional-ness, and is only ever non-None once validation passed.
            candidate: T | None = None
            verdict: Verdict
            try:
                parsed = self.parse(raw)
            except ParseError as exc:
                verdict = Verdict.invalid(f'Could not parse the response: {exc}')
            else:
                candidate = parsed
                verdict = self.validate(parsed, context)

            record = ProvenanceRecord(
                capability=self.name,
                model=config.model,
                endpoint_host=config.endpoint_host,
                prompt_sha256=prompt_digest(messages),
                raw_output=raw,
                ok=verdict.ok,
                verdict='ok' if verdict.ok else '; '.join(verdict.errors),
                attempt=attempt,
                duration_ms=duration_ms,
            )
            sink.write(record)
            provenance_ids.append(record.id)

            if verdict.ok:
                return CapabilityResult(
                    ok=True,
                    value=candidate,
                    attempts=attempt,
                    provenance_ids=tuple(provenance_ids),
                    errors=tuple(failures),
                )

            failures.extend(verdict.errors)

        return CapabilityResult(
            ok=False,
            value=None,
            attempts=self.max_attempts,
            provenance_ids=tuple(provenance_ids),
            errors=tuple(failures),
        )
