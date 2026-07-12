# Yui AI Assistant — v0.1.2

Backend da Yui: assistente pessoal de IA com usuários autenticados, memória
confiável (Redis como cache + PostgreSQL como fonte de verdade), personalidade
configurável e camada abstrata de modelos de linguagem (Claude / OpenAI).

## Arquitetura

```
app/
  core/        Configurações (fail-fast), segurança (JWT/bcrypt), personalidade, exceções
  agents/      Yui Core — orquestra cada turno em 3 fases (contexto → LLM → persistência)
  memory/      Memória de curto prazo (Redis, cache)
  models/      SQLAlchemy: users, conversations, messages, memórias, uso
  services/    LLM (abstração + provedores), memórias, histórico (rehidratação),
               contexto, rate limiting, contabilidade de uso
  api/         Rotas HTTP, schemas e dependências (auth via Bearer token)
  database/    Sessão async do PostgreSQL (pool configurável) e cliente Redis
alembic/       Migrations versionadas (fonte de verdade do schema em produção)
tests/         Unitários + integração de API (SQLite em memória, dublês de Redis/LLM)
```

Fluxo de uma mensagem: API (usuário do token) → YuiCore → *fase 1:* conversa +
memórias + histórico (Redis, com rehidratação do PostgreSQL) → *fase 2:*
personalidade (estável) + memórias delimitadas → LLMProvider **sem conexão de
banco aberta** → *fase 3:* persistência com `sequence` determinística +
registro de tokens/custo → cache atualizado.

## Segurança

- **Autenticação JWT** — o usuário é identificado exclusivamente pelo token;
  nenhuma rota aceita `user_id` do cliente.
- **Isolamento** — conversas e memórias têm FK para `users`; acessos cruzados
  respondem 404 sem revelar existência.
- **Rate limiting** — mensagens/minuto e orçamento diário de tokens por
  usuário (Redis), com limites por plano preparados para expansão.
- **Prompt injection** — memórias entram no prompt delimitadas como dados.
- **Erros** — detalhes ficam no log; o cliente recebe mensagens genéricas.

## Requisitos

- Python 3.11+
- Docker (para PostgreSQL e Redis) ou instâncias locais

## Executando localmente

```bash
# 1. Infraestrutura (PostgreSQL com pgvector + Redis com AOF)
docker compose up -d

# 2. Ambiente Python
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 3. Configuração
cp .env.example .env
# edite .env: ANTHROPIC_API_KEY (ou OPENAI_API_KEY) e, fora de development,
# JWT_SECRET_KEY. A aplicação NÃO SOBE com configuração essencial ausente.

# 4. Schema do banco
alembic upgrade head

# 5. Subir a API
uvicorn app.main:app --reload
```

API completa em container: `docker compose --profile app up --build`.

Documentação interativa: http://localhost:8000/docs

## Endpoints principais

| Método | Rota                          | Auth | Descrição                              |
|--------|-------------------------------|------|----------------------------------------|
| POST   | /api/v1/auth/register         | —    | Criar conta                            |
| POST   | /api/v1/auth/login            | —    | Obter token JWT                        |
| GET    | /api/v1/auth/me               | ✅   | Perfil do usuário autenticado          |
| POST   | /api/v1/chat                  | ✅   | Conversar com a Yui                    |
| POST   | /api/v1/memories              | ✅   | Salvar memória de longo prazo          |
| GET    | /api/v1/memories              | ✅   | Listar memórias próprias               |
| DELETE | /api/v1/memories/{memory_id}  | ✅   | Apagar memória própria                 |
| GET    | /health                       | —    | Liveness (processo vivo)               |
| GET    | /health/ready                 | —    | Readiness (503 se Postgres/Redis fora) |

### Exemplo

```bash
curl -s -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "leo@example.com", "password": "senha-segura-1", "name": "Leo"}'

TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "leo@example.com", "password": "senha-segura-1"}' | jq -r .access_token)

curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message": "O que você sabe sobre meus estudos?"}'
```

## Personalidade

Editável em `app/config/personality.yaml` (nome, objetivo, estilo, regras e
limitações). Validada no boot — YAML inválido impede o start. O caminho é
resolvido relativo à raiz do backend, independente do CWD.

## Migrations

```bash
alembic upgrade head                                  # aplicar
alembic revision --autogenerate -m "descrição"        # criar nova
```

Em desenvolvimento, `Base.metadata.create_all` roda no startup por
conveniência; produção usa exclusivamente as migrations.

## Qualidade

```bash
pytest              # testes (unitários + integração de API)
ruff check app tests alembic
mypy
```

## Roadmap

- **v0.2** — embeddings + pgvector, RAG, extração automática de memórias
  (ponto de troca: `MemoryService.retrieve_relevant`, sem mudança de contrato),
  streaming SSE (ponto de extensão: `LLMProvider.generate_stream`), CI.
- **v0.3** — tool use no contrato do provider, agentes (Memory/Guardian),
  voz (STT/TTS).
- **v0.4** — interface avançada e avatar.
