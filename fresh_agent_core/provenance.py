"""
Provenance recording.

Every attempt is recorded, including the ones that failed validation. In a
scientific pipeline a nondeterministic component without an audit trail is not
defensible, so the log *is* the evidence.

What is deliberately **not** recorded:

- the prompt body (a SHA-256 is stored instead; prompts can embed user data)
- credentials, in any form
- the full endpoint URL, which can carry credentials in userinfo or query params

Opt in to storing prompt bodies with ``record_prompts=True`` when the inputs are
known to be safe and you are debugging.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

ENV_LOG_PATH = 'FRESH_AGENT_LOG'
DEFAULT_LOG_PATH = Path('.fresh-agent') / 'provenance.jsonl'


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prompt_digest(messages: list[dict[str, str]]) -> str:
    """
    Stable SHA-256 over a message list.

    Sorted keys so that the digest depends on content rather than dict ordering,
    which makes it usable for spotting repeated prompts across runs.
    """
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class ProvenanceRecord:
    """One model call. Written whether or not it succeeded."""

    capability: str
    model: str
    endpoint_host: str
    prompt_sha256: str
    raw_output: str
    ok: bool
    verdict: str
    attempt: int
    duration_ms: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_utc_now)
    prompt: list[dict[str, str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form. Omits ``prompt`` unless it was captured."""
        data = asdict(self)
        if data.get('prompt') is None:
            data.pop('prompt', None)
        return data


@runtime_checkable
class ProvenanceSink(Protocol):
    """Somewhere provenance records go."""

    def write(self, record: ProvenanceRecord) -> None:
        """Persist *record*."""
        ...


class JSONLSink:
    """
    Append records to a JSON Lines file.

    One JSON object per line, so a partially written file stays readable and
    ``tail -f`` works during a long run.

    :param path: Destination. Defaults to ``$FRESH_AGENT_LOG``, else
        ``./.fresh-agent/provenance.jsonl``.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            env_path = os.environ.get(ENV_LOG_PATH)
            path = Path(env_path) if env_path else DEFAULT_LOG_PATH
        self.path = Path(path)

    def write(self, record: ProvenanceRecord) -> None:
        """Append one record, creating parent directories as needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')


class MemorySink:
    """
    Collect records in memory.

    For tests, and for callers that want to inspect an interaction without
    touching disk.
    """

    def __init__(self) -> None:
        self.records: list[ProvenanceRecord] = []

    def write(self, record: ProvenanceRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def attempts(self) -> int:
        """Number of recorded attempts, successful or not."""
        return len(self.records)


class NullSink:
    """Discard records. For callers who explicitly do not want an audit trail."""

    def write(self, record: ProvenanceRecord) -> None:
        return None
