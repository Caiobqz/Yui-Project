# Yui AI Assistant — v0.2

Backend da Yui: inteligência pessoal com memória semântica (RAG), extração
automática de memórias, agentes especializados, ferramentas (tool calling),
compactação de contexto e streaming — sobre a base autenticada e testada
da v0.1.2.

## Arquitetura

```
app/
  core/        Configurações (fail-fast), segurança (JWT/bcrypt), personalidade,
               exceções, background (pós-turno)
  agents/      YuiCore + MemoryAgent, PlannerAgent, ResearchAgent, TaskAgent,
               GuardianAgent
  tools/       Ferramentas expostas ao modelo: tarefas, notas, memórias,
               planner, busca web (registry + validação de schema)
  memory/      Memória de curto prazo (Redis, cache)
  models/      SQLAlchemy: users, conversations, messages, memórias (pgvector),
               tasks (planos/etapas), notes, uso
  services/    LLM (abstração + Claude/OpenAI, tools e streaming), embeddings,
               memórias, histórico, resumo, rate limiting, contabilidade
  api/         Rotas HTTP (REST + SSE), schemas e dependências
  database/    Sessão async do PostgreSQL (pool configurável) e cliente Redis
alembic/       Migrations versionadas (0001 schema, 0002 memória semântica)
tests/         56 testes: unitários + integração de API (SQLite em memória)
```

### Fluxo de um turno

```
Usuário → API (token JWT)
  Fase 0  rate limit + embedding da consulta        (sem banco)
  Fase 1  conversa + memórias semânticas + histórico
          (rehidratação Redis←PostgreSQL) + resumo  (conexão curta)
  Fase 2  loop agêntico: LLM ⇄ ferramentas
          Guardian valida → TaskAgent executa       (SEM conexão de banco)
  Fase 3  persiste turno + uso por chamada          (conexão curta)
  Pós-turno (background): MemoryAgent extrai memórias novas;
          ConversationSummarizer compacta conversas longas
```

### Agentes

| Agente | Responsabilidade |
|---|---|
| **YuiCore** | Entende intenção (via tool calling), coordena o turno e a resposta final |
| **MemoryAgent** | Cria (explícita e automaticamente), recupera (semântica/lexical) e deduplica memórias |
| **PlannerAgent** | Divide objetivos em etapas persistidas como plano acompanhável |
| **ResearchAgent** | Busca informações externas (DuckDuckGo; `WEB_SEARCH_ENABLED`) |
| **TaskAgent** | Executa chamadas de ferramenta validadas, sem derrubar o turno |
| **GuardianAgent** | Valida ferramentas/argumentos, bloqueia segredos em memórias, limita resultados |

### Memória semântica (RAG)

Conversa → análise pós-turno (LLM) → triagem (Guardian) → embedding →
pgvector → recuperação por similaridade de cosseno no próximo turno.
Cada memória tem conteúdo, categoria, importância, confiança, origem
(`user`/`extracted`), data de criação e última utilização. Com
`EMBEDDING_PROVIDER=disabled` (a Anthropic não oferece API de embeddings),
a recuperação cai automaticamente para busca lexical.

### Ferramentas disponíveis ao modelo

`create_task`, `list_tasks`, `complete_task`, `create_note`, `list_notes`,
`save_memory`, `create_plan` e `web_search` (opcional). Toda chamada passa
pelo GuardianAgent antes de executar; resultados voltam ao modelo até a
resposta final (máximo `LLM_MAX_TOOL_ITERATIONS` rodadas).

## Executando localmente

```bash
docker compose up -d                      # PostgreSQL (pgvector) + Redis (AOF)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                      # ANTHROPIC_API_KEY; opcional: OPENAI_API_KEY + EMBEDDING_PROVIDER=openai
alembic upgrade head
uvicorn app.main:app --reload             # docs em http://localhost:8000/docs
```

## Endpoints principais

| Método | Rota                          | Auth | Descrição                                   |
|--------|-------------------------------|------|----------------------------------------------|
| POST   | /api/v1/auth/register         | —    | Criar conta                                  |
| POST   | /api/v1/auth/login            | —    | Obter token JWT                              |
| GET    | /api/v1/auth/me               | ✅   | Perfil do usuário                            |
| POST   | /api/v1/chat                  | ✅   | Conversar (resposta completa)                |
| POST   | /api/v1/chat/stream           | ✅   | Conversar via SSE (delta/tool/done/error)    |
| GET/POST/DELETE | /api/v1/memories     | ✅   | Memórias de longo prazo                      |
| GET    | /api/v1/tasks?status=         | ✅   | Tarefas e planos (progresso)                 |
| GET    | /api/v1/notes                 | ✅   | Notas                                        |
| GET    | /health, /health/ready        | —    | Liveness / readiness                         |

### Exemplo de streaming

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message": "Me lembre de estudar Python amanhã às 18h"}'
# data: {"type":"tool","tools":["create_task"]}
# data: {"type":"delta","text":"Anotado! ..."}
# data: {"type":"done","conversation_id":"...","model":"...","memories_used":1}
```

## Qualidade

```bash
pytest                            # 56 testes
ruff check app tests alembic
mypy
```

## Roadmap

- **v0.3** — voz (STT/TTS sobre o canal SSE), roteamento de modelo barato para
  extração/resumo, índice ivfflat no pgvector, ferramentas calendar/files
  (exigem sandbox do Guardian), CI.
- **v0.4** — interface avançada e avatar.
