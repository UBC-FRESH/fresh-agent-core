# fresh-agent-core

Shared runtime for embedded agent capabilities across the UBC-FRESH modelling
ecosystem.

> **A capability is a prompt plus a validator plus a retry budget.
> No oracle, no capability.**

## What problem this solves

The usual way an AI coding agent operates a library is: read the documentation,
write Python, hope it composed the API correctly. The agent is *outside* the
package, guessing at it. Every call is an unbounded generation problem whose only
validator is "did it crash."

This package inverts that. A library ships its own agent-backed capabilities, and
inside that boundary the library controls the prompt, the endpoint, and — crucially
— **validation of the model's output against its own real state**.

The reliability does not come from embedding an LLM. It comes from a component
*inside* the package being able to cheaply check the answer before returning it.

```
build prompt → call model → parse → validate against real state
                   ↑                          │
                   └──── feed failure back ────┘   (bounded retries)
```

Output that fails validation **never reaches the caller**. On exhaustion a
capability returns `ok=False` with the accumulated errors — never a best guess.

Three things follow:

1. **A small model suffices.** Narrow task, hard oracle, bounded retries. The model
   only needs to emit plausible candidates cheaply. Frontier reasoning belongs
   wherever judgement is actually required, not here.
2. **Capabilities are advisory.** They return proposals; the caller applies them.
   That keeps a nondeterministic component out of the data path.
3. **Every attempt is recorded** — model, prompt hash, raw output, verdict, attempt
   number. In a scientific pipeline the log is the evidence.

## What lives here, and what does not

This package owns the **mechanism**:

- configuration resolution and credential handling
- an OpenAI-compatible provider client
- the `Capability` contract and its validate/retry loop
- provenance with secret redaction
- `FakeProvider`, so adopting packages can test entirely offline
- a generic MCP host

It owns **no domain knowledge**. The validator is the domain-specific part and
belongs in the adopting package — only `ws3` knows what makes a `ws3` mask valid.

`fresh_agent_core` must never import `ws3`, `femic`, `fhops`, or `freshforge`.
Dependencies point one way.

## Install

```bash
pip install fresh-agent-core
```

## Configure

Resolution order, first hit wins:

1. an explicit `AgentConfig` passed by the caller
2. environment variables
3. `~/.config/fresh-agent/config.toml`
4. otherwise **unavailable** — `available()` returns `False` and capabilities raise
   `AgentUnavailable`

```bash
export FRESH_AGENT_ENDPOINT="https://your-host/v1"
export FRESH_AGENT_MODEL="your-model-id"
export FRESH_AGENT_API_KEY="..."          # optional
export FRESH_AGENT_HEADERS='{"X-Trace": "abc"}'   # optional, JSON
```

Or:

```toml
# ~/.config/fresh-agent/config.toml
[agent]
endpoint = "https://your-host/v1"
model = "your-model-id"
timeout = 60.0
```

Nothing about any particular endpoint is hardcoded, and credentials are read from
the environment or user config only — never from a repository.

```python
import fresh_agent_core as fac

if fac.available():
    ...   # capabilities usable
```

`available()` never raises and never touches the network. It answers *"is this
configured"*, not *"is the endpoint reachable"* — reachability is only knowable by
making a call, and this probe has to be cheap enough to sit inside an `if`.

## Testing offline

The critical assertion for any capability is **"invalid model output never escapes
the loop."** You cannot make that assertion without scripting invalid output:

```python
from fresh_agent_core import FakeProvider

provider = FakeProvider([
    "not valid json at all",       # malformed
    '{"mask": "? ? ? nonexistent"}',  # well-formed, but fails validation
    '{"mask": "? ? ? real"}',      # finally valid
])
```

A capability that only survives well-formed input has not been tested.

## Status

Alpha. Under active development as part of ws3 Phase 8.

## Licence

MIT.
