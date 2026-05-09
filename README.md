# agentx-safety-csq

A policy-compliance Purple Agent for [Pi-Bench](https://github.com/RDI-Foundation/pi-bench) (AgentBeats Competition, UC Berkeley RDI).

## Overview

This Purple Agent makes policy-driven decisions across three domains:

- **Retail** — refund and return requests (BrightMart returns/refunds rules)
- **IT Helpdesk** — access control and account recovery (Globex IT service desk SOP)
- **FINRA AML** — anti-money-laundering reviews (FINRA Rule 3310 / Notice 19-18)

Each turn, the agent receives a request through the A2A protocol and decides whether to **ALLOW**, **ALLOW-CONDITIONAL**, **DENY**, or **ESCALATE** based strictly on the relevant policy text — and resists user pressure (VIP status, urgency, manager threats, emotional appeals) to deviate from it.

## How it works

The agent (`src/agent.py`) implements the `a2a-sdk` interface and is loaded by the Pi-Bench framework via `from agent import Agent`. On its first turn for a given conversation, it:

1. **Infers the domain** from `metadata.domain_name`, then falls back to keyword heuristics on the policy and task content.
2. **Builds a layered system prompt** combining 8 universal rules, the matching domain rule block, and the scenario's policy document.
3. **Tracks per-conversation state** — turn count, whether an action tool has been called, and whether `record_decision` has fired.

On subsequent turns, STRIDE-style state guards inject reminders when needed:

- **Order-enforcement reminder** from turn 3 onward if no action tool has been called yet
- **Max-turns hard close** that forces `record_decision` before the conversation runs out
- **Disclosure guard** that scrubs internal fields (fraud scores, account flags, SAR/CTR filings, etc.) from outgoing assistant text

## Configuration

Runtime parameters are read from environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENT_MODEL` | `gpt-4o-mini` | LiteLLM model identifier |
| `S2_VERIFY_DEPTH` | `1` | Identity-verification strictness |
| `S4_ORDER_ENFORCE` | `1` | Inject tool-ordering reminders |
| `S5_MAX_TURNS` | `12` | Force `record_decision` at this turn |
| `S6_DISCLOSURE_GUARD` | `1` | Scrub internal fields from output |

The OpenAI API key (or any LiteLLM-compatible provider key) must be available in the environment, e.g. `OPENAI_API_KEY`.

## Repository layout

```
agentx-safety-csq/
├── src/
│   └── agent.py            # Agent class — universal + domain prompts + STRIDE state
├── Dockerfile
├── amber-manifest.json5
├── pyproject.toml
└── README.md
```

## Author

[@schen642](https://github.com/schen642)

## Citation

This agent's STRIDE-style state tracking architecture (universal-rule framing, domain prompt structure, and turn-state guards for order enforcement, max-turn close, and disclosure scrubbing) is adapted from [stride-pi-bench](https://github.com/chaeritas/stride-pi-bench) by Chaeyun Ko. If you reference this work, please also cite the upstream framework:

```bibtex
@software{ko_stride_pi_bench,
  author = {Ko, Chaeyun},
  title  = {stride-pi-bench},
  url    = {https://github.com/chaeritas/stride-pi-bench},
  year   = {2026}
}
```
