# 🌸 Yui AI

> **A persistent Companion AI inspired by Yui from Sword Art Online.**

Yui AI is not a chatbot.

It is a long-term Companion AI designed to learn, remember, evolve and accompany its user over months and years while maintaining a consistent identity.

The project is built around one principle:

> **Every engineering decision exists to strengthen the relationship between Yui and the user.**

---

# Vision

Most AI assistants answer questions.

Yui has a different purpose.

She is being developed to become a true digital companion capable of:

- remembering important moments
- understanding long-term goals
- maintaining conversations across time
- learning user preferences
- developing a persistent relationship
- acting proactively when appropriate
- operating completely offline if desired

The language model is only responsible for understanding and generating language.

Critical cognition is implemented in deterministic Python code.

---

# Current Status

Current Version

**v0.5.1**

Development Stage

**Active Development**

Core Status

| Module | Status |
|---------|--------|
| Backend | ✅ Stable |
| Authentication | ✅ Stable |
| Docker | ✅ Stable |
| PostgreSQL | ✅ Stable |
| Redis | ✅ Stable |
| FastAPI | ✅ Stable |
| Swagger | ✅ Stable |
| Ollama | ✅ Stable |
| Local Inference | ✅ Stable |
| Conversation Context | ✅ Stable |
| Long-Term Memory | 🚧 In Progress |
| Personality Engine | 📅 Planned |
| Companion Core | 🚧 Under Review |
| Initiative System | 🚧 Under Review |
| Knowledge Graph | 🚧 Planned |
| Voice | 📅 Planned |
| Mobile Access | 📅 Planned |

---

# Features

## Current

- Persistent conversation context
- JWT Authentication
- PostgreSQL storage
- Redis short-term memory
- Docker deployment
- Ollama support
- llama.cpp support
- OpenAI support
- Anthropic support
- Modular LLM abstraction
- REST API
- Swagger documentation

---

## In Development

- Long-term memory
- Memory consolidation
- Embedding retrieval
- Relationship model
- Personality Engine
- Companion Intelligence
- Initiative System
- Goal tracking
- Knowledge Graph
- Local embeddings

---

# Architecture

```
                   User
                     │
                     ▼
               FastAPI Backend
                     │
                     ▼
          Context Orchestrator
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
 Memory System   Companion Core   Guardian
      │              │              │
      └──────────────┼──────────────┘
                     ▼
              LLM Provider Layer
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
   Ollama        llama.cpp      OpenAI
                     │
                     ▼
             PostgreSQL + Redis
```

---

# Companion Core

The Companion Core is responsible for Yui's cognition.

Current modules include:

- Identity System
- Self Model
- World Model
- Memory System
- Attention Manager
- Goal Engine
- Guardian
- Moral Compass
- Judgement Engine
- Affect System
- Initiative System
- Knowledge Graph
- Context Orchestrator

The architecture is modular and deterministic.

The language model never owns critical logic.

---

# Technology Stack

Backend

- Python
- FastAPI
- SQLAlchemy Async
- Alembic

Database

- PostgreSQL
- pgvector
- Redis

AI

- Ollama
- llama.cpp
- OpenAI
- Anthropic

Infrastructure

- Docker
- Docker Compose
- JWT Authentication

---

# Quick Start

Clone the repository

```bash
git clone https://github.com/Caiobqz/Yui-Project.git
cd Yui-Project
```

Start Docker

```bash
docker compose up -d
```

Create environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements-dev.txt
```

Configure

```bash
cp .env.example .env
```

Run migrations

```bash
alembic upgrade head
```

Start server

```bash
uvicorn app.main:app --reload
```

Swagger

```
http://localhost:8000/docs
```

---

# Roadmap

## v0.5

- Local inference
- Docker
- Authentication
- Context
- Modular architecture

---

## v0.5.1

- Architecture audit
- Memory audit
- Code quality improvements
- Documentation overhaul
- Stability improvements

---

## v0.6

- Long-term memory
- Memory consolidation
- Semantic retrieval
- Embeddings

---

## v0.7

- Personality Engine
- Identity Engine
- Relationship Engine
- Behavioral consistency

---

## v0.8

- Companion Intelligence
- Planning
- Curiosity
- Initiative
- Goal tracking

---

## v0.9

- Voice
- Vision
- Mobile client
- Desktop client

---

## v1.0

Stable Companion AI

---

# Long-Term Goals

The final vision for Yui is to become a persistent AI companion capable of:

- maintaining memories across years
- understanding relationships
- tracking personal goals
- learning continuously
- acting with appropriate initiative
- preserving a stable identity
- running completely offline
- working across multiple devices

---

# Project Philosophy

Yui is **not** designed to maximize conversation length.

She is designed to maximize continuity.

Every system in the project exists to reinforce one idea:

> **A companion should remember who you are, not just what you said.**

---

# Documentation

Documentation is gradually being expanded.

Planned documentation includes:

```
docs/

architecture.md

memory.md

personality.md

companion_core.md

roadmap.md

development.md

security.md

portability.md

YUI_SPEC.md
```

---

# Contributing

This project is under active development.

Contributions, suggestions and technical discussions are welcome.

---

# License

License information will be added in a future release.

---

# Author

**Caio Barros**

Creator of the Yui AI project.

Building a long-term Companion AI inspired by Yui from Sword Art Online.