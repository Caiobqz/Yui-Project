# Yui AI Assistant — v0.1

Backend da Yui: assistente pessoal de IA com memória, personalidade configurável
e camada abstrata de modelos de linguagem (Claude / OpenAI).

## Arquitetura

```
app/
  core/        Configurações e personalidade (YAML validado por Pydantic)
  agents/      Yui Core — agente que orquestra cada turno de conversa
  memory/      Memória de curto prazo (Redis)
  models/      Modelos SQLAlchemy (conversas, mensagens, memórias)
  services/    Camada de LLM (abstração + provedores), memória de longo prazo, contexto
  api/         Rotas HTTP, schemas e dependências
  database/    Sessão async do PostgreSQL e cliente Redis
alembic/       Migrations
tests/         Testes unitários
```

Fluxo de uma mensagem: API → YuiCore → (Redis: histórico) + (PostgreSQL: memórias
relevantes) → contexto (personalidade + memórias) → LLMProvider → resposta →
persistência.

## Requisitos

- Python 3.11+
- Docker (para PostgreSQL e Redis) ou instâncias locais

## Executando localmente

```bash
# 1. Infraestrutura
docker compose up -d

# 2. Ambiente Python
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configuração
cp .env.example .env
# edite .env e adicione sua ANTHROPIC_API_KEY (ou OPENAI_API_KEY)

# 4. Subir a API (em desenvolvimento as tabelas são criadas automaticamente)
uvicorn app.main:app --reload
```

Documentação interativa: http://localhost:8000/docs

## Endpoints principais

| Método | Rota                                   | Descrição                          |
|--------|----------------------------------------|------------------------------------|
| POST   | /api/v1/chat                           | Conversar com a Yui                |
| POST   | /api/v1/memories                       | Salvar memória de longo prazo      |
| GET    | /api/v1/memories/{user_id}             | Listar memórias                    |
| DELETE | /api/v1/memories/{user_id}/{memory_id} | Apagar memória                     |
| GET    | /health                                | Saúde da app, PostgreSQL e Redis   |

### Exemplo

```bash
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"user_id": "leo", "category": "estudos", "content": "Está aprendendo Python e IA", "relevance": 0.9}'

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "leo", "message": "O que você sabe sobre meus estudos?"}'
```

## Personalidade

Editável em `app/config/personality.yaml` (nome, objetivo, estilo, regras e
limitações). Validada no boot da aplicação — YAML inválido impede o start.

## Migrations (produção)

```bash
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

Em desenvolvimento, `Base.metadata.create_all` roda no startup por conveniência.

## Testes

```bash
pytest
```

## Roadmap

- **v0.2** — embeddings + pgvector, RAG, extração automática de memórias
  (o ponto de troca é `MemoryService.retrieve_relevant`, sem mudança de contrato)
- **v0.3** — voz (Speech-to-Text / Text-to-Speech)
- **v0.4** — interface avançada e avatar
