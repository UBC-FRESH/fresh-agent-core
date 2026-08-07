"""Tests for provenance recording, its sinks, and the registry."""

from __future__ import annotations

import json

import pytest

from fresh_agent_core.capability import Capability, Verdict
from fresh_agent_core.provenance import (
    DEFAULT_LOG_PATH,
    ENV_LOG_PATH,
    JSONLSink,
    MemorySink,
    NullSink,
    ProvenanceRecord,
    ProvenanceSink,
    prompt_digest,
)
from fresh_agent_core.registry import Registry


def _record(**overrides) -> ProvenanceRecord:
    defaults = {
        'capability': 'demo',
        'model': 'test-model',
        'endpoint_host': 'example.test',
        'prompt_sha256': '0' * 64,
        'raw_output': '{"answer": "x"}',
        'ok': True,
        'verdict': 'ok',
        'attempt': 1,
        'duration_ms': 12.5,
    }
    defaults.update(overrides)
    return ProvenanceRecord(**defaults)


class TestPromptDigest:
    def test_is_a_sha256_hex_digest(self):
        digest = prompt_digest([{'role': 'user', 'content': 'hello'}])
        assert len(digest) == 64
        assert set(digest) <= set('0123456789abcdef')

    def test_is_stable_for_identical_content(self):
        messages = [{'role': 'user', 'content': 'hello'}]
        assert prompt_digest(messages) == prompt_digest(list(messages))

    def test_differs_for_different_content(self):
        a = prompt_digest([{'role': 'user', 'content': 'one'}])
        b = prompt_digest([{'role': 'user', 'content': 'two'}])
        assert a != b

    def test_is_insensitive_to_key_ordering(self):
        """Depends on content, not dict iteration order, so it is comparable across runs."""
        a = prompt_digest([{'role': 'user', 'content': 'x'}])
        b = prompt_digest([{'content': 'x', 'role': 'user'}])
        assert a == b


class TestProvenanceRecord:
    def test_gets_an_id_and_timestamp_automatically(self):
        record = _record()
        assert record.id
        assert record.timestamp

    def test_ids_are_unique(self):
        assert _record().id != _record().id

    def test_timestamp_is_utc_iso8601(self):
        assert '+00:00' in _record().timestamp

    def test_serialises_to_json(self):
        payload = json.loads(json.dumps(_record().to_dict()))
        assert payload['capability'] == 'demo'
        assert payload['model'] == 'test-model'

    def test_prompt_body_omitted_when_not_captured(self):
        """Prompts can embed user data, so the digest is the default record."""
        assert 'prompt' not in _record().to_dict()

    def test_prompt_body_included_when_explicitly_captured(self):
        record = _record(prompt=[{'role': 'user', 'content': 'debugging'}])
        assert record.to_dict()['prompt'][0]['content'] == 'debugging'


class TestNoSecretsInRecords:
    """
    Provenance must never become a credential leak.

    Records carry the endpoint *host* rather than the full URL precisely because a
    URL can hold credentials in userinfo or query parameters.
    """

    def test_record_has_no_field_for_credentials(self):
        fields = set(_record().to_dict())
        for forbidden in ('api_key', 'headers', 'authorization', 'endpoint'):
            assert forbidden not in fields

    def test_host_only_never_full_url(self):
        record = _record(endpoint_host='example.test')
        serialised = json.dumps(record.to_dict())
        assert 'https://' not in serialised
        assert '/v1' not in serialised

    def test_written_jsonl_contains_no_credentials(self, tmp_path):
        sink = JSONLSink(tmp_path / 'p.jsonl')
        sink.write(_record())
        text = (tmp_path / 'p.jsonl').read_text()
        for forbidden in ('Bearer ', 'api_key', 'Authorization', 'secret'):
            assert forbidden not in text


class TestSinks:
    def test_memory_sink_collects(self):
        sink = MemorySink()
        sink.write(_record())
        sink.write(_record())
        assert sink.attempts == 2

    def test_memory_sink_is_falsy_when_empty(self):
        """
        Documents the trap.

        MemorySink defines __len__, so an empty one is falsy. Any code doing
        `sink or default` would silently discard a caller's sink on the first
        call, when it is empty by definition. The capability loop uses
        `if sink is None` for this reason.
        """
        assert not MemorySink()
        assert MemorySink() is not None

    def test_null_sink_discards(self):
        assert NullSink().write(_record()) is None

    def test_jsonl_sink_appends_one_object_per_line(self, tmp_path):
        path = tmp_path / 'p.jsonl'
        sink = JSONLSink(path)
        sink.write(_record(attempt=1))
        sink.write(_record(attempt=2))

        lines = path.read_text().strip().split('\n')
        assert len(lines) == 2
        assert [json.loads(ln)['attempt'] for ln in lines] == [1, 2]

    def test_jsonl_sink_creates_parent_directories(self, tmp_path):
        path = tmp_path / 'nested' / 'deeper' / 'p.jsonl'
        JSONLSink(path).write(_record())
        assert path.is_file()

    def test_jsonl_sink_honours_env_var(self, tmp_path, monkeypatch):
        target = tmp_path / 'from-env.jsonl'
        monkeypatch.setenv(ENV_LOG_PATH, str(target))
        assert JSONLSink().path == target

    def test_jsonl_sink_default_path(self, monkeypatch):
        monkeypatch.delenv(ENV_LOG_PATH, raising=False)
        assert JSONLSink().path == DEFAULT_LOG_PATH

    @pytest.mark.parametrize('sink_factory', [MemorySink, NullSink, lambda: JSONLSink('x')])
    def test_all_sinks_satisfy_the_protocol(self, sink_factory):
        assert isinstance(sink_factory(), ProvenanceSink)


class _Demo(Capability[str]):
    name = 'demo'
    description = 'Demo capability, validated against a fixed answer.'

    def build_messages(self, inputs, failures):
        return [{'role': 'user', 'content': str(inputs)}]

    def parse(self, raw):
        return raw.strip()

    def validate(self, candidate, context):
        return Verdict.valid() if candidate == 'ok' else Verdict.invalid('not ok')


class _Other(_Demo):
    name = 'other'
    description = 'Another one.'


class TestRegistry:
    def test_registers_and_retrieves(self):
        registry = Registry([_Demo()])
        assert registry.get('demo').name == 'demo'

    def test_len_and_contains(self):
        registry = Registry([_Demo(), _Other()])
        assert len(registry) == 2
        assert 'demo' in registry
        assert 'absent' not in registry

    def test_names_are_sorted(self):
        assert Registry([_Other(), _Demo()]).names() == ['demo', 'other']

    def test_duplicate_names_are_rejected(self):
        """
        Silently replacing would let a tool call dispatch somewhere other than
        where the caller read the description.
        """
        registry = Registry([_Demo()])
        with pytest.raises(ValueError, match='already registered'):
            registry.register(_Demo())

    def test_unknown_name_lists_what_is_available(self):
        """Usually hit by an agent that guessed, so the error should orient it."""
        registry = Registry([_Demo(), _Other()])
        with pytest.raises(KeyError, match='demo, other'):
            registry.get('nope')

    def test_unknown_name_on_empty_registry_is_explicit(self):
        with pytest.raises(KeyError, match='none registered'):
            Registry().get('nope')

    def test_describe_returns_names_and_descriptions(self):
        described = Registry([_Demo()]).describe()
        assert described == [
            {'name': 'demo', 'description': 'Demo capability, validated against a fixed answer.'}
        ]

    def test_is_iterable(self):
        assert {c.name for c in Registry([_Demo(), _Other()])} == {'demo', 'other'}
