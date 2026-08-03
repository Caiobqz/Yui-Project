# Yui AI Companion — v0.5 (Cognitive Evolution & Companion Behavior)

A Yui é uma companheira digital: observa, aprende, lembra, compreende, protege
e orienta o usuário ao longo do tempo. Seu propósito não é responder perguntas
— é cuidar do usuário. Toda decisão de engenharia serve a esse propósito.

**Realismo:** a Yui não possui consciência nem emoções humanas reais. Seus
afetos, curiosidade e iniciativas são estados computacionais PERSISTENTES —
genuínos como modelo, mas não humanos. Identidade e comportamento são
implementados em CÓDIGO; o LLM é usado para compreender e gerar linguagem,
nunca para lógica crítica.

## Princípio de engenharia

> Sempre que uma decisão puder ser determinística, ela NÃO depende do modelo
> de linguagem. Toda lógica crítica (atenção, objetivos, julgamento moral,
> permissões, self model) é Python puro e testável.

## Arquitetura cognitiva

```
Yui Cognitive Core
  ├── Identity System        cognition/identity.py — imutável, em código
  ├── Self Model             cognition/self_model.py — read-only (quem é, versão,
  │                          capacidades, ferramentas, módulos, saúde)
  ├── World Model            cognition/world_model.py — 4 domínios nunca misturados:
  │                          self · user · environment · general
  ├── Memory System          hierárquica (semantic/episodic/procedural/relationship),
  │                          consolidação, decaimento, poda
  ├── Attention Manager      cognition/attention.py — Attention Score determinístico
  │                          (similaridade · importância · recência · objetivos ·
  │                          relacionamento · preferência · grafo − redundância)
  ├── Goal Engine            cognition/goal_engine.py — progresso, parado,
  │                          abandonado, concluído; termos de objetivos ativos
  ├── Affective State        cognition/affect.py — afetos computacionais
  │                          persistentes (apego/alegria/preocupação), com
  │                          decaimento por meia-vida; influenciam prompt e
  │                          julgamento (v0.5)
  ├── Initiative System      services/initiative_service.py — iniciativas
  │                          persistidas: raras (teto), nunca repetitivas
  │                          (cooldown por situação), entregues no próximo
  │                          momento natural de conversa (v0.5)
  ├── Knowledge Graph        cognition/knowledge_graph.py — grafo derivado
  │                          (usuário → objetivos → etapas → categorias)
  ├── Moral Compass          cognition/moral_compass.py — julga ações autônomas por
  │                          benefício/risco/reversibilidade/urgência/alinhamento
  ├── Judgement Engine       cognition/judgement.py — Percepção→…→Decisão→Aprendizado
  ├── Permission System      services/permission_service.py — portão rígido de
  │                          ferramentas (ações pedidas pelo usuário)
  ├── Context Orchestrator   services/context_orchestrator.py — ÚNICO montador do
  │                          prompt, com orçamento de tokens por bloco
  └── Observability          core/metrics.py — decisões, atenção, raciocínio, moral
```

### Fluxo de um turno

```
Entrada → contexto emocional (heurística)
        → Context Orchestrator:
             Attention Manager → Memory → Goals → Relationship →
             Affective State → Iniciativa pendente → World Model →
             Self Model → Identity → Prompt Builder
        → LLM ⇄ ferramentas (Guardian + Permission System)
        → persistência + estado afetivo + entrega de iniciativa + métricas
Pós-turno (modelo utilitário barato): memórias + adaptação; manutenção;
resumo; geração de iniciativas (determinística)
```

### Autonomia por prudência (Judgement Engine, v0.5)

Ações **iniciadas pela própria Yui** não passam por permissão binária: passam
por julgamento. O Goal Engine detecta situações (abandono, oportunidade); cada
candidata vira uma `ProposedAction` avaliada pela Bússola Moral — score e
confiança determinísticos que consideram benefício, risco, reversibilidade,
urgência, alinhamento aos objetivos, **vínculo do relacionamento** e
**preocupação do estado afetivo**, com o Permission System como portão duro.

Aprovadas (`proceed`) são PERSISTIDAS (tabela `initiatives`) com as garantias
comportamentais da companheira: raras (teto de pendentes), nunca repetitivas
(cooldown por situação), oportunas (a pendente mais antiga entra como diretiva
no próximo turno — o modelo decide se o momento é natural; entregue, não é
reoferecida). `GET /api/v1/initiatives` observa o registro, read-only.

O Self Model é **imutável**: nenhum usuário, agente ou LLM o altera.

## Endpoints novos (v0.4, read-only, autenticados)

| Rota | Descrição |
|------|-----------|
| `GET /api/v1/self` | Self Model (identidade, capacidades, ferramentas, saúde) |
| `GET /api/v1/metrics` | Observabilidade (contadores e amostras agregados) |
| `GET /api/v1/initiatives` | Iniciativas registradas (aprovadas pela Bússola Moral) |
| `GET /api/v1/knowledge-graph` | Grafo de entidades derivado do usuário |

## Executando localmente

```bash
docker compose up -d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                      # ANTHROPIC_API_KEY etc.
alembic upgrade head
uvicorn app.main:app --reload
```

## Qualidade

```bash
pytest              # 142 testes
ruff check app tests alembic
mypy
```

## Preparação para próximas versões

- **Voz (v0.5+):** SSE já é o transporte token a token do TTS; STT vira rota que
  desemboca em `stream_message`. Iniciativas pendentes ganham um canal de push
  (hoje são entregues no próximo momento natural de conversa).
- **Avatar / visão (v0.5):** eventos SSE já carregam estado para reagir; visão
  entra como novo tipo de conteúdo em `ChatMessage`.
- **Knowledge Graph persistido:** hoje é derivado; a versão futura persiste o
  grafo e extrai entidades/relações com o modelo utilitário.
- **Ferramentas sensíveis (arquivos, SO, calendário, apps):** nascem
  `default_allowed=False` — o Permission System (ações do usuário) e a Bússola
  Moral (ações autônomas) já as tratam com segurança.
