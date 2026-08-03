"""Agentes especializados da Yui.

O YuiCore (yui_core.py) coordena o turno; o roteamento de intenção acontece
via tool calling — o modelo escolhe a ferramenta e cada agente é o executor
por trás dela:

    YuiCore ─── entende intenção (tool calling) e coordena a resposta
      ├── MemoryAgent    — cria/recupera/deduplica memórias (RAG)
      ├── PlannerAgent   — divide objetivos em etapas acompanháveis
      ├── ResearchAgent  — busca informações externas
      ├── TaskAgent      — executa chamadas de ferramenta validadas
      └── GuardianAgent  — valida ações e protege privacidade
"""
