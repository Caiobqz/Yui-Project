"""Yui Cognitive Core — o núcleo cognitivo da Yui (v0.3).

Inspiração: uma companheira analítica, curiosa, protetora e consistente.
Implementação realista: a Yui NÃO possui consciência nem emoções humanas
reais — todos os comportamentos abaixo (incluindo afetos e iniciativas,
v0.5) são modelos computacionais, determinísticos ou derivados de chamadas
de modelo.

    Yui Cognitive Core
      ├── Identity System        cognition/identity.py      (imutável, em código)
      ├── Memory System          models/memory.py + services/memory_service.py
      │                          + services/memory_maintenance.py
      │                          (working/episódica/semântica/procedural/relacionamento,
      │                           consolidação, reforço, decaimento, poda)
      ├── Personality Engine     core/personality.py (traços/estilo fixos, YAML)
      ├── Reasoning Engine       cognition/reasoning.py     (sinais → estratégia)
      ├── Curiosity Engine       cognition/curiosity.py     (lacunas → 1 pergunta)
      ├── Adaptation Engine      cognition/user_model.py    (perfil aprendido)
      ├── Emotional Context      cognition/emotional_context.py (heurística)
      ├── Planning System        agents/planner_agent.py + tools/planner.py
      ├── Action System          tools/ + agents/task_agent.py
      └── Permission System      services/permission_service.py + GuardianAgent

O YuiCore (agents/yui_core.py) orquestra o fluxo cognitivo do turno:
entrada → contexto emocional → memória relevante → modelo do usuário →
estratégia → resposta/ação; pós-turno: análise (TurnAnalyzer, modelo
utilitário) → novas memórias + adaptação + manutenção + resumo.
"""
