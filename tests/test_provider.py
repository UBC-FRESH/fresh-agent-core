"""Tests for the provider client and the offline test double."""

from __future__ import annotations

import pytest

from fresh_agent_core import AgentConfig, FakeProvider, OpenAIProvider, Provider
from fresh_agent_core.errors import ProviderError
from fresh_agent_core.provider import _extract_content


def _config(**kwargs) -> AgentConfig:
    return AgentConfig(endpoint='https://example.test/v1', model='m', **kwargs)


class TestProviderProtocol:
    def test_openai_provider_satisfies_protocol(self):
        assert isinstance(OpenAIProvider(_config()), Provider)

    def test_fake_provider_satisfies_protocol(self):
        """
        The double must be substitutable for the real client.

        If it drifts from the protocol, offline tests stop proving anything about
        the online path.
        """
        assert isinstance(FakeProvider(['hello']), Provider)


class TestResponseExtraction:
    """
    Malformed provider responses must be reported as provider problems.

    Otherwise they surface later as a confusing TypeError or KeyError from inside a
    capability's parser, which points the reader at the wrong layer.
    """

    def test_extracts_content(self):
        body = {'choices': [{'message': {'content': 'hello'}}]}
        assert _extract_content(body, 'host') == 'hello'

    @pytest.mark.parametrize('body, fragment', [
        ('not a dict', 'expected a JSON object'),
        ({}, "no 'choices'"),
        ({'choices': []}, "no 'choices'"),
        ({'choices': ['not a dict']}, "no 'message' object"),
        ({'choices': [{}]}, "no 'message' object"),
        ({'choices': [{'message': {}}]}, 'expected a string'),
        ({'choices': [{'message': {'content': 42}}]}, 'expected a string'),
    ])
    def test_malformed_bodies_raise_provider_error(self, body, fragment):
        with pytest.raises(ProviderError, match=fragment):
            _extract_content(body, 'host')

    def test_error_names_the_host(self):
        with pytest.raises(ProviderError, match='example.test'):
            _extract_content({}, 'example.test')

    def test_error_does_not_leak_full_url(self):
        """Provenance and errors carry the host, never a URL that may hold credentials."""
        cfg = AgentConfig(endpoint='https://user:pw@example.test/v1', model='m')
        with pytest.raises(ProviderError) as exc:
            _extract_content({}, cfg.endpoint_host)
        assert '/v1' not in str(exc.value)


class TestFakeProvider:
    def test_replays_scripted_responses_in_order(self):
        provider = FakeProvider(['first', 'second'])
        assert provider.complete([]) == 'first'
        assert provider.complete([]) == 'second'

    def test_raises_scripted_exceptions(self):
        provider = FakeProvider([ProviderError('simulated outage')])
        with pytest.raises(ProviderError, match='simulated outage'):
            provider.complete([])

    def test_exhaustion_is_an_explicit_error(self):
        """
        Running past the script is a test bug, and should say so.

        Silently repeating would let a test claim a retry budget was respected when
        it was never exercised.
        """
        provider = FakeProvider(['only one'])
        provider.complete([])
        with pytest.raises(ProviderError, match='exhausted'):
            provider.complete([])

    def test_repeat_last_keeps_returning_final_response(self):
        provider = FakeProvider(['always'], repeat_last=True)
        assert [provider.complete([]) for _ in range(5)] == ['always'] * 5

    def test_rejects_empty_script(self):
        with pytest.raises(ValueError, match='at least one'):
            FakeProvider([])

    def test_records_calls_for_inspection(self):
        """
        Recorded prompts are what let a test prove failures were fed back.

        Without this, a retry loop that silently re-sent the identical prompt would
        be indistinguishable from one that incorporated the validation failure.
        """
        provider = FakeProvider(['a', 'b'])
        provider.complete([{'role': 'user', 'content': 'first prompt'}])
        provider.complete([{'role': 'user', 'content': 'second prompt'}])

        assert provider.call_count == 2
        assert provider.calls[0][0]['content'] == 'first prompt'
        assert provider.last_prompt()[0]['content'] == 'second prompt'

    def test_last_prompt_is_none_before_any_call(self):
        assert FakeProvider(['x']).last_prompt() is None

    def test_recorded_calls_are_snapshots(self):
        """Mutating the caller's list afterwards must not rewrite history."""
        provider = FakeProvider(['x'])
        messages = [{'role': 'user', 'content': 'original'}]
        provider.complete(messages)
        messages.append({'role': 'user', 'content': 'added later'})

        assert len(provider.calls[0]) == 1


class TestNoNetworkOnImport:
    def test_importing_the_package_performs_no_network_io(self, monkeypatch):
        """
        `import fresh_agent_core` must never touch the network.

        Adopting packages import this at module scope, so an accidental request
        here would make `import ws3` depend on connectivity.
        """
        import importlib
        import socket

        def _forbidden(*args, **kwargs):
            raise AssertionError('import performed network I/O')

        monkeypatch.setattr(socket.socket, 'connect', _forbidden)
        monkeypatch.setattr(socket, 'create_connection', _forbidden)

        import fresh_agent_core
        importlib.reload(fresh_agent_core)

    def test_httpx_is_not_imported_eagerly(self):
        """
        httpx is imported lazily inside complete().

        Keeps `import fresh_agent_core` cheap for callers that only ever check
        availability and never make a request.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    'import sys, fresh_agent_core; '
                    'print("httpx" in sys.modules)'
                ),
            ],
            capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == 'False'
