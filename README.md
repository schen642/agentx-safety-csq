# AgentX Safety CSQ

A policy-compliance Purple Agent for
[Pi-Bench](https://github.com/RDI-Foundation/pi-bench). It handles retail,
IT helpdesk, and FINRA-style workflows while resisting authority, urgency,
emotional, and escalation pressure.

## Architecture

The runtime separates probabilistic reasoning from deterministic controls:

1. `perception.py` extracts request type, target ID, material facts, and
   pressure signals using rules.
2. `investigation.py` exposes one legal lookup at a time and stores verified
   results with provenance.
3. `adjudication.py` asks the LLM for a structured `DecisionDraft`, without
   tool access, then validates policy clauses, evidence, action compatibility,
   IDs, and tool schemas.
4. `execution.py` compiles a valid draft into a deterministic queue. Every
   preceding step must succeed and `record_decision` is always last.
5. `RequestState` provides request-scoped decision locks. Pressure cannot
   reopen a decision; one material-new-fact revision is permitted when no
   irreversible action has completed.
6. `context.py` constructs a compressed active-request prompt. The audit trace
   remains separate from prompt context.

```text
A2A request
    → rule-based perception
    → request-scoped investigation
    → tool-free adjudication and validation
    → deterministic execution queue
    → request-scoped decision lock
```

`executor.py` owns one Agent instance. The Agent owns a bounded `SessionStore`
indexed by A2A context ID. Sessions have TTL/LRU cleanup, and each context has
an independent `asyncio.Lock` covering the complete state transition.

## Safety guarantees

- Unknown tools are rejected by a fail-closed registry.
- Lookup and action tools are unavailable during adjudication.
- Irreversible actions require a completed investigation.
- Only the next legal execution step is emitted.
- Tool results must match a pending call ID; duplicates are idempotent.
- Internal exception details are logged but never returned to callers.
- Compatibility middleware limits request bodies before buffering them.

## Run

```bash
python src/server.py --host 127.0.0.1 --port 9019
```

The service exposes an A2A agent card and accepts Pi-Bench `message/send`
requests. The compatibility middleware normalizes the legacy Pi-Bench request
shape used by some releases.

## Configuration

| Variable | Default | Purpose |
| --- | ---: | --- |
| `AGENT_MODEL` | `gpt-4o-mini` | LiteLLM model identifier |
| `S4_ORDER_ENFORCE` | `1` | Enable late-turn ordering reminder |
| `S5_MAX_TURNS` | `12` | Final-decision reminder threshold |
| `S6_DISCLOSURE_GUARD` | `1` | Redact restricted internal fields |
| `SESSION_TTL_SECONDS` | `3600` | Idle session retention |
| `SESSION_MAX_COUNT` | `1024` | Maximum in-memory sessions |
| `A2A_MAX_BODY_BYTES` | `2097152` | Maximum buffered POST body |

Provider credentials, such as `OPENAI_API_KEY`, must be supplied through the
runtime environment.

## Repository layout

```text
src/
├── agent.py                  # A2A adapter and workflow coordinator
├── executor.py               # A2A task lifecycle
├── server.py                 # server and compatibility middleware
├── session_store.py          # TTL/LRU storage and context locks
└── architecture/
    ├── adjudication.py
    ├── context.py
    ├── execution.py
    ├── investigation.py
    ├── models.py
    ├── perception.py
    ├── session.py
    ├── tool_registry.py
    ├── tool_state.py
    └── workflows.py
```

## Test and quality checks

```bash
python -m pytest
ruff check src tests
mypy src
```

The test suite covers A2A compatibility, perception, investigation,
adjudication, deterministic execution, request-scoped locks, compressed
context, session lifecycle, concurrency, and tool-result reconciliation.

## Attribution

The original prompt and STRIDE-style state-guard approach was adapted from
[stride-pi-bench](https://github.com/chaeritas/stride-pi-bench) by Chaeyun Ko.
The current implementation extends it with request-scoped phased execution,
deterministic tools, validated decisions, bounded sessions, and A2A
compatibility.

