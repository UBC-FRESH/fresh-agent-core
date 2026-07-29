"""
Capability registry.

A named collection of capabilities. Adopting packages assemble one and hand it to
the MCP host, which turns each entry into a tool. Keeping discovery in one place
is what lets the host stay generic and domain-free.
"""

from __future__ import annotations

from typing import Any, Iterator

from fresh_agent_core.capability import Capability


class Registry:
    """
    A collection of capabilities, keyed by name.

    :param capabilities: Optional initial entries.
    """

    def __init__(self, capabilities: Any = None) -> None:
        self._capabilities: dict[str, Capability[Any]] = {}
        for capability in capabilities or ():
            self.register(capability)

    def register(self, capability: Capability[Any]) -> None:
        """
        Add *capability*.

        :raises ValueError: If the name is already taken. Silently replacing would
            mean a tool call could dispatch somewhere other than where the caller
            read the description.
        """
        if capability.name in self._capabilities:
            raise ValueError(
                f'A capability named {capability.name!r} is already registered. '
                f'Names must be unique: MCP tool dispatch is keyed on them.'
            )
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> Capability[Any]:
        """
        Look up a capability by name.

        :raises KeyError: With the available names, since this is usually hit by
            an agent that guessed.
        """
        try:
            return self._capabilities[name]
        except KeyError:
            available = ', '.join(sorted(self._capabilities)) or '(none registered)'
            raise KeyError(
                f'No capability named {name!r}. Available: {available}'
            ) from None

    def names(self) -> list[str]:
        """Registered names, sorted."""
        return sorted(self._capabilities)

    def describe(self) -> list[dict[str, str]]:
        """
        Name and description for each capability.

        This is what an external agent reads to decide what to call, so the
        descriptions should say what each capability *validates*.
        """
        return [
            {'name': c.name, 'description': c.description}
            for c in sorted(self._capabilities.values(), key=lambda c: c.name)
        ]

    def __len__(self) -> int:
        return len(self._capabilities)

    def __contains__(self, name: object) -> bool:
        return name in self._capabilities

    def __iter__(self) -> Iterator[Capability[Any]]:
        return iter(self._capabilities.values())
