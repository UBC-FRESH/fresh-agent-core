"""
Test doubles.

Exported as part of the public API rather than hidden in a test directory, because
every adopting package needs to test its own capabilities offline. The point is not
convenience -- it is that the critical assertion for any capability is
**"invalid model output never escapes the loop"**, and you cannot make that
assertion without being able to script invalid output on demand.
"""

from __future__ import annotations

from collections.abc import Iterable

from fresh_agent_core.errors import ProviderError


class FakeProvider:
    """
    A :py:class:`~fresh_agent_core.provider.Provider` that replays scripted responses.

    Each call to :py:meth:`complete` returns the next scripted item. An item may be:

    - a ``str`` -- returned as the completion
    - an ``Exception`` -- raised, to simulate provider failure

    Scripting malformed and plausible-but-wrong completions is the intended use.
    A capability that only survives well-formed input has not been tested.

    :param responses: Scripted responses, consumed in order.
    :param repeat_last: If True, keep returning the final response once exhausted
        instead of raising. Useful when asserting that a retry budget is respected
        without having to script exactly the right number of failures.
    """

    def __init__(
        self,
        responses: Iterable[str | Exception],
        *,
        repeat_last: bool = False,
    ) -> None:
        self._responses: list[str | Exception] = list(responses)
        if not self._responses:
            raise ValueError('FakeProvider needs at least one scripted response.')
        self._repeat_last = repeat_last
        self._index = 0
        #: Every message list this provider was called with, in order. Lets tests
        #: assert that validation failures were actually fed back into the next
        #: prompt, rather than the loop silently retrying with the same input.
        self.calls: list[list[dict[str, str]]] = []

    @property
    def call_count(self) -> int:
        """How many times :py:meth:`complete` has been called."""
        return len(self.calls)

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the next scripted response, or raise it if it is an exception."""
        self.calls.append(list(messages))

        if self._index >= len(self._responses):
            if not self._repeat_last:
                raise ProviderError(
                    f'FakeProvider exhausted: {len(self._responses)} response(s) were '
                    f'scripted but call {self.call_count} was made. Script more '
                    f'responses, or pass repeat_last=True.'
                )
            item = self._responses[-1]
        else:
            item = self._responses[self._index]
            self._index += 1

        if isinstance(item, Exception):
            raise item
        return item

    def last_prompt(self) -> list[dict[str, str]] | None:
        """The most recent message list, or None if never called."""
        return self.calls[-1] if self.calls else None
