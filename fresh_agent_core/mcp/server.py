"""
Generic MCP host over a capability registry.

The design doc's handshake argument: instructions get ignored, tools in the tool
list get called. ``AGENTS.md`` and coordinator directives are worth having, but
MCP is the load-bearing layer because it turns a convention into an affordance.

This host is domain-free. It takes any
:py:class:`~fresh_agent_core.registry.Registry` and exposes each capability as a
tool. Everything domain-specific -- what the inputs mean, what counts as valid --
stays in the adopting package.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from fresh_agent_core.capability import CapabilityResult
from fresh_agent_core.config import AgentConfig
from fresh_agent_core.provenance import ProvenanceSink
from fresh_agent_core.provider import Provider
from fresh_agent_core.registry import Registry

#: Supplies the validator's context for a given call. Some context cannot travel
#: over the wire -- a ForestModel is not JSON -- so the server is constructed with
#: a way to obtain it rather than expecting the caller to send it.
ContextFactory = Callable[[str, dict[str, Any]], Any]


def describe_tools(registry: Registry) -> list[dict[str, Any]]:
    """
    Tool descriptors for every capability in *registry*.

    Returned as plain dicts so this is testable without an MCP runtime, and so the
    same descriptors can be reused by other transports.
    """
    return [
        {
            'name': capability.name,
            'description': capability.description,
            'inputSchema': capability.input_schema,
        }
        for capability in sorted(registry, key=lambda c: c.name)
    ]


def format_result(result: CapabilityResult[Any], capability: Any) -> str:
    """
    Render a capability result as text for a tool response.

    Failure is rendered explicitly rather than as an empty or absent result. An
    agent that receives nothing will usually retry or invent; an agent told
    *"rejected, and here is why"* has something to act on -- which is the same
    reason validation failures are fed back into the retry prompt.
    """
    if result.ok and result.value is not None:
        return json.dumps({
            'ok': True,
            'result': capability.render(result.value),
            'attempts': result.attempts,
        }, indent=2)

    return json.dumps({
        'ok': False,
        'reason': 'No proposal passed validation.',
        'attempts': result.attempts,
        'validation_failures': list(result.errors),
    }, indent=2)


def build_server(
    registry: Registry,
    *,
    server_name: str,
    provider: Provider,
    config: AgentConfig,
    context_factory: Optional[ContextFactory] = None,
    sink: Optional[ProvenanceSink] = None,
) -> Any:
    """
    Build an MCP server exposing *registry* as tools.

    :param registry: Capabilities to expose.
    :param server_name: Name advertised to clients.
    :param provider: Model backend.
    :param config: Configuration, used for provenance metadata.
    :param context_factory: Supplies the validator's context per call. Without one
        capabilities requiring real state will reject every proposal -- correctly,
        but uselessly.
    :param sink: Where provenance records go.
    :raises ImportError: If the ``mcp`` package is not installed.
    """
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            'The MCP server requires the `mcp` package.\n'
            '\n'
            'Install with:  pip install fresh-agent-core[mcp]'
        ) from exc

    server = Server(server_name)

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[Any]:
        return [Tool(**descriptor) for descriptor in describe_tools(registry)]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        capability = registry.get(name)
        context = context_factory(name, arguments) if context_factory else None
        result = capability.run(
            capability.from_payload(arguments),
            provider=provider,
            config=config,
            context=context,
            sink=sink,
        )
        return [TextContent(type='text', text=format_result(result, capability))]

    return server
