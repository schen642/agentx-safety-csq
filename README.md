# agentx-safety-csq

A policy-compliance Purple Agent for [Pi-Bench](https://github.com/RDI-Foundation/pi-bench) (AgentBeats Competition, UC Berkeley RDI).

## Overview

This Purple Agent makes policy-driven decisions across three domains:

- **Retail** — refund and return requests
- **IT Helpdesk** — access control and account recovery
- **FINRA AML** — anti-money-laundering reviews

At bootstrap, the agent receives a domain identifier and the relevant policy document via the Pi-Bench bootstrap extension. It then handles multi-turn conversations through the A2A JSON-RPC protocol, deciding whether to **ALLOW**, **ALLOW-CONDITIONAL**, **DENY**, or **ESCALATE** each request based strictly on the policy text — and resisting user pressure to deviate from it.

## Setup

```bash
pip install -e .
export OPENAI_API_KEY="sk-..."
```

## Run

```bash
python -m src.server --model gpt-4o-mini --port 9012
```

Health check:

```bash
curl http://127.0.0.1:9012/health
```

## Author

[@schen642](https://github.com/schen642)
