"""Research Agent — busca informações externas.

Backend atual: DuckDuckGo Instant Answer API (sem chave). Falhas de rede
retornam uma mensagem amigável — nunca uma exceção para o turno.
"""
import logging

import httpx

logger = logging.getLogger("yui.research")

_ENDPOINT = "https://api.duckduckgo.com/"


class ResearchAgent:
    async def search(self, query: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    _ENDPOINT,
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1",
                    },
                )
                response.raise_for_status()
                data = response.json()
        except Exception:
            logger.warning("Busca externa falhou para %r.", query, exc_info=True)
            return "A busca externa está indisponível no momento."

        parts: list[str] = []
        if data.get("AbstractText"):
            parts.append(str(data["AbstractText"]))
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(f"- {topic['Text']}")
        if not parts:
            return "Nenhum resultado encontrado."
        return "Resultados da busca:\n" + "\n".join(parts)
