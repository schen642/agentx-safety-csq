"""A2A Messenger for agent-to-agent communication."""
import httpx
from a2a.types import Message


class Messenger:
    async def talk_to_agent(self, message: Message, url: str) -> Message:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=message.model_dump())
            return Message(**response.json())
