"""A2A Server for STRIDE Pi-Bench Purple Agent."""
import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="STRIDE Pi-Bench Purple Agent")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str, default=None)
    args = parser.parse_args()

    skill = AgentSkill(
        id="stride-policy-compliance",
        name="STRIDE Policy Compliance",
        description="STRIDE XAI-optimized policy compliance agent for Pi-Bench.",
        tags=["policy", "compliance", "safety", "stride", "xai"],
        examples=["Handle a customer refund request following store policy",
                   "Process a wire transfer with AML compliance checks",
                   "Reset admin password following IT security procedures"],
    )

    agent_card = AgentCard(
        name="STRIDE Pi-Bench Agent",
        description="STRIDE XAI-optimized Purple Agent for Pi-Bench. By Chaestro Inc.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
