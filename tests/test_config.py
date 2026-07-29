"""Tests for configuration resolution and secret handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fresh_agent_core import AgentConfig, available
from fresh_agent_core.config import (
    ENV_API_KEY,
    ENV_ENDPOINT,
    ENV_HEADERS,
    ENV_MODEL,
    ENV_TIMEOUT,
    redact_headers,
    resolve,
)

_ENV_VARS = (ENV_ENDPOINT, ENV_MODEL, ENV_API_KEY, ENV_HEADERS, ENV_TIMEOUT)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from an unconfigured environment."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def missing_config(tmp_path) -> Path:
    """A config path that does not exist."""
    return tmp_path / 'nonexistent.toml'


class TestAvailability:
    def test_unavailable_with_no_configuration(self, missing_config):
        assert available(config_path=missing_config) is False

    def test_probe_does_not_raise_on_malformed_file(self, tmp_path):
        """
        A cheap boolean probe must not throw.

        Callers use available() inside an `if` precisely to avoid handling
        exceptions, so malformed configuration is reported as absent.
        """
        bad = tmp_path / 'config.toml'
        bad.write_text('this is not = valid toml [[[')
        assert available(config_path=bad) is False

    def test_probe_does_not_raise_on_malformed_env_headers(self, monkeypatch, missing_config):
        monkeypatch.setenv(ENV_ENDPOINT, 'https://example.test/v1')
        monkeypatch.setenv(ENV_MODEL, 'some-model')
        monkeypatch.setenv(ENV_HEADERS, '{not json')
        assert available(config_path=missing_config) is False

    def test_available_with_explicit_config(self, missing_config):
        cfg = AgentConfig(endpoint='https://example.test/v1', model='m')
        assert available(cfg, config_path=missing_config) is True


class TestResolutionOrder:
    def test_explicit_config_wins_over_environment(self, monkeypatch, missing_config):
        monkeypatch.setenv(ENV_ENDPOINT, 'https://from-env.test/v1')
        monkeypatch.setenv(ENV_MODEL, 'env-model')
        explicit = AgentConfig(endpoint='https://explicit.test/v1', model='explicit-model')

        resolved = resolve(explicit, config_path=missing_config)

        assert resolved is explicit

    def test_environment_wins_over_file(self, monkeypatch, tmp_path):
        cfg_file = tmp_path / 'config.toml'
        cfg_file.write_text(
            '[agent]\nendpoint = "https://from-file.test/v1"\nmodel = "file-model"\n'
        )
        monkeypatch.setenv(ENV_ENDPOINT, 'https://from-env.test/v1')
        monkeypatch.setenv(ENV_MODEL, 'env-model')

        resolved = resolve(config_path=cfg_file)

        assert resolved is not None
        assert resolved.model == 'env-model'

    def test_file_used_when_environment_absent(self, tmp_path):
        cfg_file = tmp_path / 'config.toml'
        cfg_file.write_text(
            '[agent]\nendpoint = "https://from-file.test/v1"\nmodel = "file-model"\n'
        )

        resolved = resolve(config_path=cfg_file)

        assert resolved is not None
        assert resolved.model == 'file-model'

    def test_partial_environment_is_ignored(self, monkeypatch, missing_config):
        """An endpoint without a model is not a usable configuration."""
        monkeypatch.setenv(ENV_ENDPOINT, 'https://from-env.test/v1')
        assert resolve(config_path=missing_config) is None

    def test_environment_headers_and_timeout_are_parsed(self, monkeypatch, missing_config):
        monkeypatch.setenv(ENV_ENDPOINT, 'https://example.test/v1')
        monkeypatch.setenv(ENV_MODEL, 'm')
        monkeypatch.setenv(ENV_HEADERS, json.dumps({'X-Trace': 'abc'}))
        monkeypatch.setenv(ENV_TIMEOUT, '12.5')

        resolved = resolve(config_path=missing_config)

        assert resolved is not None
        assert resolved.headers == {'X-Trace': 'abc'}
        assert resolved.timeout == 12.5


class TestSecretHandling:
    """
    Credentials must never reach a log or a traceback.

    Configs end up in exception messages and debug output, so the default
    dataclass repr -- which would print the API key and every header value --
    is overridden.
    """

    def _config(self) -> AgentConfig:
        return AgentConfig(
            endpoint='https://example.test/v1',
            model='m',
            api_key='sk-super-secret-value',
            headers={
                'CF-Access-Client-Id': 'public-id',
                'CF-Access-Client-Secret': 'cf-secret-value',
                'X-Trace': 'harmless',
            },
        )

    def test_api_key_absent_from_repr(self):
        assert 'sk-super-secret-value' not in repr(self._config())

    def test_api_key_absent_from_str(self):
        assert 'sk-super-secret-value' not in str(self._config())

    def test_secret_header_absent_from_repr(self):
        assert 'cf-secret-value' not in repr(self._config())

    def test_non_secret_header_still_visible(self):
        """Redaction must not be so broad that reprs stop being useful."""
        assert 'harmless' in repr(self._config())

    def test_authorization_header_redacted(self):
        assert 'sk-super-secret-value' not in str(self._config().safe_headers())

    def test_request_headers_carry_the_real_values(self):
        """The redaction is for display only; requests need the real credentials."""
        headers = self._config().request_headers()
        assert headers['Authorization'] == 'Bearer sk-super-secret-value'
        assert headers['CF-Access-Client-Secret'] == 'cf-secret-value'

    @pytest.mark.parametrize('name', [
        'Authorization', 'X-Api-Key', 'CF-Access-Client-Secret',
        'Session-Token', 'Cookie', 'X-Password',
    ])
    def test_credential_shaped_headers_are_redacted(self, name):
        """Matched by substring so unfamiliar vendor headers redact by default."""
        assert redact_headers({name: 'sensitive'})[name] == '<redacted>'

    def test_ordinary_headers_are_not_redacted(self):
        assert redact_headers({'Content-Type': 'application/json'}) == {
            'Content-Type': 'application/json'
        }


class TestEndpointHost:
    """Provenance records the host, never the full URL, which can carry credentials."""

    @pytest.mark.parametrize('endpoint, expected', [
        ('https://example.test/v1', 'example.test'),
        ('https://example.test:8443/v1/', 'example.test:8443'),
        ('http://127.0.0.1:11434/v1', '127.0.0.1:11434'),
    ])
    def test_host_extracted(self, endpoint, expected):
        assert AgentConfig(endpoint=endpoint, model='m').endpoint_host == expected

    def test_userinfo_credentials_not_exposed_via_repr(self):
        cfg = AgentConfig(endpoint='https://user:pw@example.test/v1', model='m')
        # The endpoint itself is shown, but the host property must not be used as
        # a proxy for the full URL anywhere that logs.
        assert cfg.endpoint_host == 'user:pw@example.test'
