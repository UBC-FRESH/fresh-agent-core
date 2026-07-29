"""
Tests for the capability contract and its validate/retry loop.

The assertion that matters most here is **invalid model output never escapes the
loop**. Everything else is supporting detail. These tests script malformed
responses, plausible-but-wrong responses, and endless-failure responses, and assert
that none of them reach a caller as a successful result.
"""

from __future__ import annotations

import json

import pytest

from fresh_agent_core import AgentConfig, FakeProvider
from fresh_agent_core.capability import (
    Capability,
    CapabilityResult,
    ParseError,
    Verdict,
)
from fresh_agent_core.errors import ProviderError, ValidationExhausted
from fresh_agent_core.provenance import MemorySink

CONFIG = AgentConfig(endpoint='https://example.test/v1', model='test-model')

#: The oracle for these tests: a candidate is valid only if it appears here.
#: Stands in for "resolve this mask against a real ForestModel".
KNOWN_GOOD = {'valid-answer'}


class PickAnswer(Capability[str]):
    """A minimal capability whose oracle is membership in a known set."""

    name = 'pick_answer'
    description = 'Returns an answer, validated against a known set of real answers.'

    def build_messages(self, inputs, failures):
        content = f'Question: {inputs}'
        if failures:
            content += '\nPrevious attempts failed because: ' + '; '.join(failures)
        return [{'role': 'user', 'content': content}]

    def parse(self, raw):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParseError(f'expected a JSON object, got {raw!r} ({exc})') from exc
        if not isinstance(payload, dict) or 'answer' not in payload:
            raise ParseError("expected a JSON object with an 'answer' key")
        return str(payload['answer'])

    def validate(self, candidate, context):
        known = context if context is not None else KNOWN_GOOD
        if candidate in known:
            return Verdict.valid()
        return Verdict.invalid(f'{candidate!r} is not a known answer')


def _ok(answer: str = 'valid-answer') -> str:
    return json.dumps({'answer': answer})


def _run(responses, *, context=None, sink=None, capability=None, repeat_last=False):
    cap = capability or PickAnswer()
    provider = FakeProvider(responses, repeat_last=repeat_last)
    result = cap.run(
        'a question',
        provider=provider,
        config=CONFIG,
        context=context,
        sink=sink,
    )
    return result, provider


class TestHappyPath:
    def test_valid_first_attempt_returns_value(self):
        result, provider = _run([_ok()])
        assert result.ok is True
        assert result.value == 'valid-answer'
        assert result.attempts == 1
        assert provider.call_count == 1

    def test_result_carries_provenance_ids(self):
        sink = MemorySink()
        result, _ = _run([_ok()], sink=sink)
        assert len(result.provenance_ids) == 1
        assert result.provenance_ids[0] == sink.records[0].id

    def test_unwrap_returns_the_value(self):
        result, _ = _run([_ok()])
        assert result.unwrap() == 'valid-answer'


class TestInvalidOutputNeverEscapes:
    """The central guarantee."""

    def test_malformed_output_never_returned(self):
        result, _ = _run(['this is not json', 'still not json', '{{{'])
        assert result.ok is False
        assert result.value is None

    def test_plausible_but_wrong_output_never_returned(self):
        """
        The harder case.

        Well-formed, parseable, and confidently wrong. Only the oracle
        distinguishes it, which is exactly why a capability without one is
        worthless.
        """
        result, _ = _run([_ok('wrong'), _ok('also-wrong'), _ok('still-wrong')])
        assert result.ok is False
        assert result.value is None

    def test_failure_reports_why(self):
        result, _ = _run([_ok('wrong')], repeat_last=True)
        assert any('wrong' in e for e in result.errors)

    def test_unwrap_raises_rather_than_returning_junk(self):
        result, _ = _run([_ok('wrong')], repeat_last=True)
        with pytest.raises(ValidationExhausted):
            result.unwrap()

    def test_recovers_when_a_later_attempt_is_valid(self):
        result, provider = _run(['not json', _ok('wrong'), _ok()])
        assert result.ok is True
        assert result.value == 'valid-answer'
        assert result.attempts == 3
        assert provider.call_count == 3


class TestRetryBudget:
    def test_budget_is_respected_exactly(self):
        result, provider = _run([_ok('wrong')], repeat_last=True)
        assert provider.call_count == PickAnswer.max_attempts
        assert result.attempts == PickAnswer.max_attempts

    def test_budget_is_configurable(self):
        class OneShot(PickAnswer):
            name = 'one_shot'
            max_attempts = 1

        result, provider = _run([_ok('wrong')], repeat_last=True, capability=OneShot())
        assert provider.call_count == 1
        assert result.ok is False

    def test_stops_early_on_success(self):
        _, provider = _run([_ok(), _ok(), _ok()])
        assert provider.call_count == 1


class TestFailureFeedback:
    """
    A retry that re-sends the identical prompt is a re-roll, not a repair.

    These assert the loop actually threads validation failures back into the next
    prompt, which is the difference between the two.
    """

    def test_failures_are_fed_into_the_next_prompt(self):
        _, provider = _run([_ok('wrong'), _ok()])
        second_prompt = provider.calls[1][0]['content']
        assert 'wrong' in second_prompt

    def test_first_prompt_has_no_failures(self):
        _, provider = _run([_ok()])
        assert 'failed because' not in provider.calls[0][0]['content']

    def test_prompts_differ_between_attempts(self):
        _, provider = _run([_ok('wrong-one'), _ok('wrong-two'), _ok()])
        contents = [c[0]['content'] for c in provider.calls]
        assert len(set(contents)) == 3

    def test_parse_errors_are_also_fed_back(self):
        _, provider = _run(['not json at all', _ok()])
        assert 'parse' in provider.calls[1][0]['content'].lower()


class TestProviderFailuresPropagate:
    """
    Transport failure is not a content failure.

    Retrying content cannot fix an unreachable endpoint, so it propagates rather
    than being absorbed into ok=False, which would misreport infrastructure
    trouble as a model shortcoming.
    """

    def test_provider_error_is_raised_not_swallowed(self):
        cap = PickAnswer()
        provider = FakeProvider([ProviderError('endpoint unreachable')])
        with pytest.raises(ProviderError, match='unreachable'):
            cap.run('q', provider=provider, config=CONFIG)

    def test_provider_error_is_distinct_from_validation_exhaustion(self):
        result, _ = _run([_ok('wrong')], repeat_last=True)
        assert isinstance(result, CapabilityResult)
        assert result.ok is False


class TestProvenance:
    def test_a_record_is_written_for_every_attempt_including_failures(self):
        sink = MemorySink()
        _run([_ok('wrong'), _ok('wrong'), _ok()], sink=sink)
        assert sink.attempts == 3

    def test_failed_attempts_are_recorded_as_failed(self):
        sink = MemorySink()
        _run([_ok('wrong'), _ok()], sink=sink)
        assert [r.ok for r in sink.records] == [False, True]

    def test_records_capture_the_verdict_reason(self):
        sink = MemorySink()
        _run([_ok('wrong')], repeat_last=True, sink=sink)
        assert 'not a known answer' in sink.records[0].verdict

    def test_records_carry_attempt_numbers(self):
        sink = MemorySink()
        _run([_ok('wrong'), _ok('wrong'), _ok()], sink=sink)
        assert [r.attempt for r in sink.records] == [1, 2, 3]

    def test_records_carry_model_and_host(self):
        sink = MemorySink()
        _run([_ok()], sink=sink)
        assert sink.records[0].model == 'test-model'
        assert sink.records[0].endpoint_host == 'example.test'

    def test_prompt_body_is_not_recorded_by_default(self):
        """Prompts can embed user data; a digest is stored instead."""
        sink = MemorySink()
        _run([_ok()], sink=sink)
        record = sink.records[0]
        assert record.prompt is None
        assert len(record.prompt_sha256) == 64

    def test_running_without_a_sink_is_allowed(self):
        cap = PickAnswer()
        result = cap.run('q', provider=FakeProvider([_ok()]), config=CONFIG)
        assert result.ok is True


class TestContext:
    def test_context_is_passed_to_the_validator(self):
        """
        The oracle validates against caller-supplied real state.

        Here a different context makes a previously-invalid answer valid, proving
        the validator consults it rather than a hardcoded set.
        """
        result, _ = _run([_ok('context-specific')], context={'context-specific'})
        assert result.ok is True

    def test_context_can_invalidate_a_normally_valid_answer(self):
        result, _ = _run([_ok('valid-answer')], context={'something-else'}, repeat_last=True)
        assert result.ok is False


class TestVerdict:
    def test_invalid_requires_a_reason(self):
        """
        A failing verdict with no reason gives the retry nothing to act on.

        That silently degrades the loop into repeated identical sampling, which
        looks like a retry budget but is not one.
        """
        with pytest.raises(ValueError, match='at least one reason'):
            Verdict.invalid()

    def test_valid_verdict_has_no_errors(self):
        assert Verdict.valid().ok is True
        assert Verdict.valid().errors == ()

    def test_invalid_verdict_keeps_reasons(self):
        verdict = Verdict.invalid('first', 'second')
        assert verdict.ok is False
        assert verdict.errors == ('first', 'second')


class TestSubclassContract:
    def test_capability_without_a_name_is_rejected(self):
        """
        Provenance and MCP dispatch are keyed on the name.

        Enforced at instantiation rather than subclass creation, so an
        intermediate abstract subclass may omit it.
        """
        class Nameless(PickAnswer):
            name = ''

        with pytest.raises(TypeError, match='non-empty `name`'):
            Nameless()

    def test_abstract_methods_must_be_implemented(self):
        with pytest.raises(TypeError):
            Capability()  # type: ignore[abstract]
