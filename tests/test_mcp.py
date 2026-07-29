"""
Tests for the generic MCP host.

Tool descriptors and result rendering are tested as plain data, so the guarantees
hold without an MCP runtime and without a network.
"""

from __future__ import annotations

import json

import pytest

from fresh_agent_core.capability import Capability, CapabilityResult, Verdict
from fresh_agent_core.mcp import describe_tools, format_result
from fresh_agent_core.registry import Registry


class Alpha(Capability[str]):
    name = 'alpha'
    description = 'Does alpha things. Validated against a known set.'
    input_schema = {
        'type': 'object',
        'properties': {'query': {'type': 'string'}},
        'required': ['query'],
    }

    def build_messages(self, inputs, failures):
        return [{'role': 'user', 'content': str(inputs)}]

    def parse(self, raw):
        return raw.strip()

    def validate(self, candidate, context):
        return Verdict.valid() if candidate == 'good' else Verdict.invalid('not good')

    def from_payload(self, payload):
        return payload['query']

    def render(self, value):
        return f'rendered:{value}'


class Beta(Alpha):
    name = 'beta'
    description = 'Does beta things. Validated the same way.'


class Plain(Capability[str]):
    """Overrides nothing optional, so it exercises the defaults."""

    name = 'plain'
    description = 'Uses the default schema and payload handling.'

    def build_messages(self, inputs, failures):
        return [{'role': 'user', 'content': str(inputs)}]

    def parse(self, raw):
        return raw.strip()

    def validate(self, candidate, context):
        return Verdict.valid()


@pytest.fixture
def registry():
    return Registry([Beta(), Alpha()])


class TestToolDescriptors:
    def test_one_descriptor_per_capability(self, registry):
        assert len(describe_tools(registry)) == 2

    def test_descriptors_are_sorted_by_name(self, registry):
        assert [d['name'] for d in describe_tools(registry)] == ['alpha', 'beta']

    def test_descriptor_carries_description_and_schema(self, registry):
        alpha = describe_tools(registry)[0]
        assert 'Validated' in alpha['description']
        assert alpha['inputSchema']['required'] == ['query']

    def test_descriptors_are_json_serialisable(self, registry):
        """They cross a wire, so anything unserialisable is a latent runtime failure."""
        json.dumps(describe_tools(registry))

    def test_default_schema_accepts_an_object(self):
        assert describe_tools(Registry([Plain()]))[0]['inputSchema'] == {'type': 'object'}


class TestPayloadMapping:
    def test_from_payload_converts_tool_arguments(self):
        assert Alpha().from_payload({'query': 'hello'}) == 'hello'

    def test_default_from_payload_passes_the_dict_through(self):
        assert Plain().from_payload({'a': 1}) == {'a': 1}


class TestResultRendering:
    def test_successful_result_is_rendered(self):
        result = CapabilityResult(
            ok=True, value='good', attempts=1, provenance_ids=(), errors=(),
        )
        payload = json.loads(format_result(result, Alpha()))
        assert payload['ok'] is True
        assert payload['result'] == 'rendered:good'
        assert payload['attempts'] == 1

    def test_failure_is_rendered_explicitly_with_reasons(self):
        """
        An agent given nothing will retry or invent.

        Told 'rejected, and here is why', it has something to act on -- the same
        reason validation failures are fed back into the retry prompt.
        """
        result = CapabilityResult(
            ok=False, value=None, attempts=3, provenance_ids=(),
            errors=('not good', 'still not good'),
        )
        payload = json.loads(format_result(result, Alpha()))
        assert payload['ok'] is False
        assert payload['validation_failures'] == ['not good', 'still not good']
        assert payload['attempts'] == 3

    def test_failure_never_reports_a_value(self):
        """The central guarantee, restated at the transport boundary."""
        result = CapabilityResult(
            ok=False, value=None, attempts=2, provenance_ids=(), errors=('nope',),
        )
        assert 'result' not in json.loads(format_result(result, Alpha()))

    def test_rendered_output_is_json(self):
        result = CapabilityResult(
            ok=True, value='good', attempts=1, provenance_ids=(), errors=(),
        )
        json.loads(format_result(result, Alpha()))


class TestBuildServer:
    def test_requires_the_mcp_package(self, registry, monkeypatch):
        """A missing optional extra must name itself, not surface as ModuleNotFoundError."""
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name.startswith('mcp'):
                raise ImportError('no mcp')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _blocked)

        from fresh_agent_core import AgentConfig, FakeProvider
        from fresh_agent_core.mcp import build_server

        with pytest.raises(ImportError, match=r'fresh-agent-core\[mcp\]'):
            build_server(
                registry,
                server_name='test',
                provider=FakeProvider(['x']),
                config=AgentConfig(endpoint='offline://test', model='m'),
            )

    def test_builds_a_server_when_mcp_is_available(self, registry):
        pytest.importorskip('mcp')
        from fresh_agent_core import AgentConfig, FakeProvider
        from fresh_agent_core.mcp import build_server

        server = build_server(
            registry,
            server_name='test-server',
            provider=FakeProvider(['good'], repeat_last=True),
            config=AgentConfig(endpoint='offline://test', model='m'),
        )
        assert server is not None
