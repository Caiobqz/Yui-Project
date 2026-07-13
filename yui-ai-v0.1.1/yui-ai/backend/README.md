# Yui AI Assistant — v0.3 (Cognitive Core)

Backend da Yui: inteligência pessoal com identidade permanente, memória
hierárquica com consolidação e esquecimento, adaptação por usuário,
curiosidade funcional, contexto emocional, agentes, ferramentas com sistema
de permissões e streaming.

**Realismo:** a Yui não possui consciência, emoções reais nem vontade
própria. Todos os comportamentos do núcleo cognitivo são modelos
computacionais de interação — e a própria identidade declara isso ao modelo.

## Yui Cognitive Core

```
Yui Cognitive Core
  ├── Identity System        cognition/identity.py — imutável, em código
  ├── Memory System          working (Redis+resumo) | semantic | episodic |
  │                          procedural | relationship; consolidação por
  │                          reforço, decaimento por meia-vida, poda
  ├── Personality Engine     traços/estilo (YAML) — QUEM é ≠ COMO conversa
  ├── Reasoning Engine       cognition/reasoning.py — sinais → estratégia
  ├── Curiosity Engine       lacunas (objetivos desconhecidos, planos parados)
  │                          → no máximo 1 pergunta sugerida por turno
  ├── Adaptation Engine      notas aprendidas por usuário ("prefere exemplos
  │                          práticos"), com dedupe e teto
  ├── Emotional Context      heurística: frustração/dificuldade/pressa/
  │                          motivação → modula tom e complexidade
  ├── Planning System        criar, acompanhar (get_plan_progress) e revisar
  │                          (review_plan) objetivos
  ├── Action System          ferramentas validadas pelo Guardian
  └── Permission System      autorização por usuário/ferramenta; categorias
                             sensíveis futuras nascem negadas
```

### Fluxo cognitivo de um turno

```
Entrada → contexto emocional (heurística)
        → memórias relevantes (similaridade × importância × recência)
        → modelo do usuário (adaptação + relacionamento) + permissões
        → curiosidade (lacunas) → estratégia → system prompt
        → LLM ⇄ ferramentas (Guardian + permissões; TaskAgent executa)
        → persistência + interação registrada
Pós-turno (modelo utilitário barato, em background):
        TurnAnalyzer → memórias tipadas + notas de adaptação
        → manutenção periódica (esquecimento) → resumo de conversas longas
```

### Memória hierárquica

| Camada | Onde | Exemplo |
|---|---|---|
| Working | Redis + `conversations.summary` | contexto atual |
| Semantic | `memory_entries` (type=semantic) | "gosta de programação" |
| Episodic | type=episodic (meia-vida 30d) | "terminou o projeto X" |
| Procedural | type=procedural | "prefere commits pequenos" |
| Relationship | type=relationship + `user_profiles` | marcos e continuidade |

Cada memória tem importância, confiança, origem, data, frequência de uso e
última utilização. Reconfirmações **reforçam** a memória existente
(consolidação); memórias extraídas, nunca usadas e irrelevantes são
**podadas** — as criadas pelo usuário, nunca.

## Executando localmente

```bash
docker compose up -d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                      # ANTHROPIC_API_KEY etc.
alembic upgrade head
uvicorn app.main:app --reload             # docs em http://localhost:8000/docs
```

## Endpoints principais

| Método | Rota                              | Descrição                                  |
|--------|-----------------------------------|---------------------------------------------|
| POST   | /api/v1/auth/{register,login}     | Conta e token JWT                           |
| POST   | /api/v1/chat                      | Conversar (resposta completa)               |
| POST   | /api/v1/chat/stream               | Conversar via SSE (delta/tool/done/error)   |
| GET/POST/DELETE | /api/v1/memories         | Memórias (com tipo, uso e confiança)        |
| GET    | /api/v1/tasks, /api/v1/notes      | Tarefas/planos e notas                      |
| GET    | /api/v1/permissions               | Permissões efetivas de ferramentas          |
| PUT    | /api/v1/permissions/{tool_name}   | Conceder/revogar uma ferramenta             |
| GET    | /health, /health/ready            | Liveness / readiness                        |

## Qualidade

```bash
pytest              # 84 testes
ruff check app tests alembic
mypy
```

## Preparação para as próximas versões

- **v0.4 (voz):** o canal SSE é o transporte do TTS token a token; STT entra
  como nova rota que desemboca no mesmo `YuiCore.stream_message`; wake word é
  responsabilidade do cliente.
- **v0.5 (avatar/visão/desktop):** eventos SSE (`delta`/`tool`/`done`) já
  carregam o que um avatar precisa para reagir; visão entraria como novo tipo
  de conteúdo no contrato `ChatMessage`.
- **Ferramentas sensíveis (arquivos, SO, calendário, apps):** registram
  `default_allowed=False` e categoria própria — o Permission System já nega
  até o usuário conceder via API.
