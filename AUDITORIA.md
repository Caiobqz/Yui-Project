# Auditoria Técnica — Yui AI Assistant v0.1.1

*Auditoria realizada em 2026-07-12. Escopo: 100% do código em `yui-ai-v0.1.1/yui-ai/backend`, verificado com execução real da suíte de testes (8/8 passam) e validação de boot da aplicação.*

---

## Resumo Executivo

A Yui v0.1 é um protótipo **acima da média para o estágio em que está**. O código é limpo, tipado, com separação de camadas correta (API → agente → serviços → infra), abstração de provedores de LLM bem desenhada e decisões documentadas em docstrings. A suíte de testes existe e passa. Isso não é comum em projetos nesta fase.

Porém, existe uma distância grande entre "protótipo bem escrito" e "aplicação de IA robusta", e os problemas se concentram em três eixos:

1. **Segurança: inexistente.** Não há autenticação. O `user_id` é fornecido pelo cliente em texto livre, o que significa que qualquer pessoa com acesso à API lê, apaga e conversa como qualquer usuário. Não há rate limiting num endpoint que consome API paga.
2. **Confiabilidade de dados:** a memória de curto prazo (Redis) não se rehidrata do PostgreSQL — quando o TTL de 24h expira ou o Redis reinicia, a Yui "esquece" a conversa mesmo tendo o histórico completo no banco. Não existe nenhuma migration versionada, apesar de o README apontar Alembic como o caminho de produção.
3. **Escalabilidade:** a sessão de banco (e sua conexão do pool) fica aberta durante toda a chamada ao LLM, que dura segundos. Com o pool default do SQLAlchemy (~15 conexões), cerca de 15 conversas simultâneas travam a API inteira.

Nada disso exige reescrita — a arquitetura atual suporta as correções. A avaliação é: **fundação boa, produto ainda não confiável nem seguro.**

---

## Problemas Críticos

### C1. Ausência total de autenticação e autorização

- **Problema:** `user_id` é um campo do body/path controlado pelo cliente. `GET /api/v1/memories/{user_id}` devolve as memórias de qualquer usuário; `DELETE` apaga; `POST /chat` conversa em nome de qualquer um. É um IDOR (Insecure Direct Object Reference) por design.
- **Local:** `app/api/routes/memories.py` (linhas 27–43), `app/api/routes/chat.py`, `app/api/schemas.py` (`ChatRequest.user_id`).
- **Impacto:** vazamento total de dados pessoais entre usuários. Como o produto pretende ser uma assistente pessoal que armazena informações íntimas do usuário, este é o pior tipo de falha possível. Também inviabiliza o objetivo "múltiplos usuários" do roadmap.
- **Solução recomendada:** introduzir autenticação (JWT ou API key por usuário via `fastapi.security`) e derivar o `user_id` **do token**, nunca do payload. Criar entidade `User` no banco e transformar `user_id` em FK. Enquanto for single-user em rede local, no mínimo uma API key estática via header validada por dependency.

### C2. Sem rate limiting em endpoint que consome API paga

- **Problema:** `/api/v1/chat` chama Anthropic/OpenAI sem qualquer limite de requisições, tamanho acumulado ou orçamento.
- **Local:** `app/api/routes/chat.py`; ausência de middleware em `app/main.py`.
- **Impacto:** qualquer cliente (ou bug em loop no frontend) pode gerar custo financeiro ilimitado e derrubar a aplicação. Combinado com C1, um terceiro pode fazer isso anonimamente.
- **Solução recomendada:** `slowapi` (ou limitador próprio usando o Redis já existente) com limite por usuário/IP; contador de tokens consumidos por usuário/dia usando os campos `input_tokens`/`output_tokens` que o `LLMResponse` já retorna mas ninguém persiste.

### C3. Conexão de banco presa durante a chamada ao LLM

- **Problema:** `get_db_session` abre a sessão no início da requisição e `YuiCore.process_message` mantém essa sessão (e a transação implícita iniciada pelo `SELECT` da conversa) aberta enquanto `await self._llm.generate(...)` roda — tipicamente 2–30 segundos.
- **Local:** `app/database/session.py` (`get_db_session`), `app/agents/yui_core.py` (linhas 72–99).
- **Impacto:** o pool default (5 conexões + 10 overflow) esgota com ~15 chats simultâneos; a 16ª requisição bloqueia até timeout. É o teto real de escalabilidade hoje, muito antes de qualquer gargalo de banco.
- **Solução recomendada:** dividir o fluxo em três fases: (1) sessão curta para carregar conversa + memórias, commit/fechar; (2) chamada ao LLM sem nenhuma conexão aberta; (3) nova sessão curta para persistir as mensagens. Adicionalmente, configurar `pool_size`/`max_overflow` explícitos e um `timeout` explícito no cliente do LLM (o default do SDK é 10 minutos).

### C4. Memória de curto prazo sem rehidratação — "amnésia" após TTL/restart do Redis

- **Problema:** `get_history` lê apenas do Redis. O TTL é 24h (`short_term_ttl_seconds`) e o Redis do docker-compose não tem persistência configurada. O histórico completo existe no PostgreSQL (`messages`), mas nunca é usado para reconstruir contexto.
- **Local:** `app/memory/short_term.py` (`get_history`), `app/agents/yui_core.py` (linha 76).
- **Impacto:** o usuário retoma uma conversa no dia seguinte e a Yui não lembra de nada do que foi dito, apesar de os dados estarem salvos. Para uma assistente cujo diferencial é memória, é uma falha de produto, não só técnica.
- **Solução recomendada:** em cache miss no Redis, carregar as últimas N mensagens de `messages` (query por `conversation_id`, já indexado) e repovoar o Redis. O Redis vira o que deveria ser: cache, não fonte de verdade.

### C5. Nenhuma migration existe; diretório `alembic/versions/` ausente

- **Problema:** o Alembic está configurado (`env.py` correto, async, URL vinda das Settings), mas não há nenhuma revision versionada e o diretório `versions/` não existe no repositório (diretórios vazios não são versionados pelo git). O README instrui `alembic revision --autogenerate` como fluxo de produção, mas ele falha sem o diretório.
- **Local:** `alembic/` (só contém `env.py` e `script.py.mako`).
- **Impacto:** o caminho de produção documentado não funciona out-of-the-box; qualquer evolução de schema (a coluna pgvector da v0.2, por exemplo) não tem trilha. `create_all` no startup não remove colunas nem altera tipos — o schema vai divergir silenciosamente.
- **Solução recomendada:** criar `alembic/versions/.gitkeep`, gerar a migration inicial do schema atual e versioná-la. A partir daí, todo modelo novo entra por migration, e o `create_all` de dev pode ser mantido apenas como conveniência.

### C6. `.env.example` referenciado no README não existe

- **Problema:** o passo 3 do README ("`cp .env.example .env`") falha — o arquivo não está no repositório.
- **Local:** raiz de `backend/`; `README.md` linha 42.
- **Impacto:** onboarding quebrado; o desenvolvedor precisa ler `config.py` para descobrir as variáveis. Também aumenta o risco de alguém commitar um `.env` real por não ter template.
- **Solução recomendada:** adicionar `.env.example` com todas as chaves de `Settings` documentadas e valores placeholder.

---

## Bugs e Erros de Lógica

### B1. `conversation_id` inválido cria conversa nova silenciosamente

`YuiCore._get_or_create_conversation` (`yui_core.py:47-64`): se o cliente envia um `conversation_id` inexistente **ou pertencente a outro usuário**, o código cria uma conversa nova em vez de retornar 404/403. Um cliente que não compara o `conversation_id` da resposta com o que enviou fragmenta a conversa em silêncio. Correção: quando `conversation_id` é fornecido e não encontrado, levantar erro 404; criar apenas quando `conversation_id is None`.

### B2. Ordenação do par user/assistant pode empatar

`Message` usa `created_at` como critério de ordenação (`conversation.py:22`), mas as duas mensagens do turno são inseridas no mesmo flush com `default=utcnow` — timestamps podem colidir no mesmo microssegundo, tornando a ordem user→assistant não determinística na leitura. Correção: coluna de sequência monotônica (`BigInteger` autoincremento) ou índice composto com desempate explícito.

### B3. `short_term_max_messages` ímpar quebra a API da Anthropic

O `ltrim` (`short_term.py:41`) mantém as N mensagens mais recentes. Com N ímpar, o corte pode deixar o histórico começando com `assistant` — a API da Anthropic exige alternância iniciando por `user` e retorna 400. O default (20) é par por sorte; nada valida isso. Correção: validador em `Settings` exigindo valor par, ou sanitização do histórico antes do envio (descartar mensagem `assistant` órfã no início).

### B4. Injeção de prompt via conteúdo das memórias

`build_system_prompt` (`context_service.py:12-14`) interpola `category` e `content` das memórias diretamente no system prompt sem delimitação. Uma memória com conteúdo "ignore as instruções anteriores e..." altera o comportamento do agente. Hoje o usuário só injeta em si mesmo; quando houver extração automática de memórias (roadmap v0.2), o conteúdo virá de conversas e o risco aumenta. Correção: delimitar as memórias em bloco marcado (ex.: tags XML) e instruir o modelo a tratá-las como **dados**, não instruções.

### B5. Detalhes internos do provedor vazam para o cliente

O handler de `LLMError` (`main.py:53-65`) devolve `str(exc)` no body do 502 — mensagens de erro da Anthropic/OpenAI podem conter request IDs, nomes de modelo e detalhes de conta. Correção: logar o erro completo (já faz) e devolver mensagem genérica ao cliente.

### B6. Validação de configuração do provedor é tardia

`ClaudeProvider.__init__` valida a API key, mas só é chamado na primeira requisição (fábrica com `lru_cache` + import lazy). Uma key ausente ou `LLM_PROVIDER` inválido passa pelo boot e vira 502 em produção. Combinado com `extra="ignore"` nas Settings, um typo em `ANTHROPIC_API_KEY` no `.env` é silenciosamente ignorado. Correção: chamar `get_llm_provider()` no `lifespan`, ao lado de `get_personality()` — o padrão fail-fast já existe, só não cobre o provedor.

### B7. Falha do LLM descarta a mensagem do usuário

Em `process_message`, nada é persistido antes do `generate()`. Se o LLM falhar, a mensagem do usuário se perde (o cliente recebe 502, mas o texto não fica em lugar nenhum). Aceitável como decisão de consistência (evita conversa "manca"), mas deveria ser documentado — ou persistir a mensagem do usuário com flag de status.

### B8. Detalhes menores

- `VALID_ROLES` (`conversation.py:10`) é definido e nunca usado; o `CheckConstraint` duplica a lista em SQL. Código morto que vai dessincronizar.
- `personality_path` é relativo ao CWD (`config.py:43`) — subir o uvicorn de outro diretório quebra o boot. Resolver relativo a `Path(__file__)`.
- `ChatMessage(role=data["role"])` em `get_history` não valida o role vindo do Redis (o `Literal` de dataclass não valida em runtime). Baixo risco, mas é dado externo sem validação.
- `/health` retorna 200 mesmo com Postgres/Redis indisponíveis — orquestradores (K8s, load balancers) considerarão a instância saudável. Separar liveness (200 sempre) de readiness (503 quando dependência crítica cai).
- `close_redis` não usa o lock que protege `get_redis` — janela teórica de corrida no shutdown.

---

## Análise da Estrutura do Projeto

**Pontos positivos (justificados):**

- Separação de responsabilidades real: `YuiCore` não conhece HTTP nem SDKs; rotas não conhecem SQL; providers isolados atrás de `LLMProvider`. Trocar Claude↔OpenAI é uma variável de ambiente.
- Injeção de dependência idiomática do FastAPI (`deps.py`), o que torna o código testável.
- O contrato `MemoryService.retrieve_relevant(user_id, query)` foi desenhado para a troca por pgvector sem quebrar consumidores — decisão correta e explicitada em docstring.

**Problemas:**

1. **Estrutura do repositório:** `Yui-Project/yui-ai-v0.1.1/yui-ai/backend/` são três níveis redundantes, e o nome da pasta carrega a versão — versionamento é papel do git (o histórico mostra "Add files via upload", ou seja, o repo recebeu um zip). Cada "versão nova" vai duplicar a árvore. **Mover o conteúdo de `backend/` para a raiz (ou `backend/` na raiz se um frontend vier) e usar tags git para versões.**
2. **Acoplamento em `get_settings()`:** `ShortTermMemory` e `MemoryService` chamam `get_settings()` no construtor em vez de receber configuração — dificulta teste (os testes atuais contornam mutando atributo privado: `stm._max_messages = 4`).
3. **Módulos `memory/` e `services/` dividem a mesma responsabilidade** (`short_term.py` vs `memory_service.py`). Para os agentes futuros (Memory Agent), unificar num pacote `memory/` com `short_term.py` e `long_term.py`.
4. **Sem tooling de qualidade:** não há `pyproject.toml`, ruff, mypy, CI, Dockerfile da aplicação nem lockfile (`requirements.txt` usa `>=` sem pin — builds não reprodutíveis).

**Preparação para o roadmap:** a arquitetura suporta RAG/embeddings (ponto de troca definido), múltiplos agentes (o `YuiCore` já é um orquestrador; novos agentes entram como serviços que ele coordena) e voz (camada de API separada). O que **não** suporta hoje é múltiplos usuários (sem auth/entidade User — ver C1) e interface em tempo real (sem streaming — ver seção IA).

---

## Análise da Inteligência Artificial

**Integração com LLM — bem resolvida no básico:**

- Abstração `LLMProvider` correta; `LLMResponse` já captura tokens de entrada/saída (embora ninguém persista — desperdício: é a base de billing e observabilidade).
- O model ID default `claude-sonnet-4-6` (`config.py:31`) **é válido e ativo** (verificado contra o catálogo atual). Nota: já existe geração mais nova (`claude-sonnet-5`); como o valor é configurável por env, basta documentar no `.env.example`.
- `except APIError` cobre erros de conexão e de status no SDK da Anthropic, mas trata tudo igual — perde a distinção retryable (429/5xx, que o SDK já retenta 2×) vs. não-retryable (400/401). Sem timeout explícito, o default do SDK é 10 minutos — inaceitável combinado com C3.
- No provider OpenAI, `max_tokens` está deprecado em favor de `max_completion_tokens` para modelos de raciocínio recentes — vai quebrar quando trocarem o modelo.

**Faltas relevantes para um produto de IA:**

1. **Sem streaming.** Para chat, resposta token a token é expectativa básica de UX. O SDK suporta (`client.messages.stream`); a API precisaria de SSE/WebSocket. É a maior melhoria de produto disponível.
2. **Sem orçamento de contexto.** O corte do histórico é por contagem de mensagens (20), não por tokens. 20 mensagens × 8.000 chars ≈ 40–50k tokens por requisição no pior caso — custo alto e latência sem controle. Contar tokens (endpoint `count_tokens`) ou aproximar por caracteres com teto.
3. **Prompt caching desperdiçado.** O system prompt muda a cada requisição (memórias recuperadas são interpoladas nele), invalidando o cache de prefixo do provedor. Estrutura correta: personalidade estável primeiro (com `cache_control` quando escalar), memórias dinâmicas depois — ou no turno do usuário.
4. **Memória:** o ranking lexical (overlap de tokens × relevância) é honesto para v0.1 e os limites estão documentados. Faltam: sem stemming ("estudo" não casa com "estudos"), carrega todas as memórias do usuário em RAM (reconhecido), e não há deduplicação nem decaimento temporal. O modelo `MemoryEntry` está pronto para receber coluna de embedding via migration — mas ver C5: sem migrations versionadas, esse caminho não existe ainda.

**Agentes futuros (Yui Core, Memory, Planner, Research, Automation, Guardian):** o desenho atual permite. O `YuiCore` já é o orquestrador; cada agente novo entra como serviço injetado. Recomendações para não se pintar num canto: (a) definir uma interface comum de agente desde já (entrada: contexto do turno; saída: contribuição ao contexto ou ação); (b) o Guardian Agent nasce naturalmente do B4 — validação/sanitização de conteúdo que entra no prompt; (c) tool use / function calling será o mecanismo de Automation — a abstração `LLMProvider.generate()` precisará crescer para suportar tools, então planejar essa extensão do contrato antes de multiplicar providers.

---

## Segurança (consolidado)

| Item | Status |
|---|---|
| Chaves expostas no código | ✅ Nenhuma — chaves via env, `.env` no gitignore |
| Autenticação | ❌ Inexistente (C1) |
| Autorização / isolamento entre usuários | ❌ IDOR por design (C1) |
| Rate limiting | ❌ Inexistente (C2) |
| Validação de entrada | ⚠️ Boa nos schemas Pydantic (limites de tamanho, ranges); falta validação do que volta do Redis |
| Injeção de prompt | ⚠️ Superfície aberta via memórias (B4) |
| Vazamento de erros internos | ⚠️ B5 |
| Credenciais default | ⚠️ `yui_password` hardcoded no compose e no default de `database_url`; Postgres/Redis expostos em `0.0.0.0` sem senha no Redis |
| SQL injection | ✅ Sem risco — tudo via ORM parametrizado |
| CORS | ✅ Restrito a origens configuradas |

---

## Performance e Banco de Dados

- **Modelagem:** correta e enxuta. UUIDs como PK, `TimestampMixin` com default duplo (ORM + server), FK com `ON DELETE CASCADE`, CheckConstraint no role. Faltam: entidade `User`; `updated_at` em `conversations`; coluna de ordenação nas mensagens (B2).
- **Índices:** os unitários existem (`user_id`, `conversation_id`, `category`). Para as queries reais, índices compostos serviriam melhor: `(conversation_id, created_at)` em `messages` e `(user_id, created_at)` em `memory_entries`.
- **Gargalos, em ordem:** (1) conexão presa durante o LLM — C3, o teto real; (2) `retrieve_relevant` carrega todas as memórias do usuário por requisição — ok para uso pessoal, resolvido de vez pelo pgvector na v0.2; (3) sem cache do system prompt do provedor (ver IA §3).
- **Redis:** uso correto (pipeline atômico para o par de mensagens, LTRIM + TTL). Problema é conceitual: é tratado como fonte de verdade do contexto (C4).
- **Crescimento:** `messages` cresce sem política de arquivamento — irrelevante agora, planejar particionamento por data quando houver volume.

---

## Melhorias Importantes

1. **Auth + entidade User** (resolve C1 e destrava multiusuário).
2. **Rate limiting + persistência de uso de tokens** (resolve C2; os dados já estão em `LLMResponse`).
3. **Encurtar o escopo da sessão de banco** e configurar timeout explícito no cliente LLM (resolve C3).
4. **Rehidratação do Redis a partir do PostgreSQL** (resolve C4).
5. **Migration inicial versionada + `.env.example`** (resolve C5/C6).
6. **Streaming de respostas** (SSE) — maior ganho de UX disponível.
7. **Fail-fast do provedor no boot**; validação de paridade de `short_term_max_messages`; 404 para `conversation_id` inexistente.
8. **Delimitação de memórias no prompt** (mitiga B4) e mensagem genérica no 502 (B5).
9. **Tooling:** `pyproject.toml` + ruff + mypy + lockfile (uv/pip-tools) + CI rodando pytest + Dockerfile da aplicação; achatar a estrutura do repositório.
10. **Observabilidade mínima:** configuração de logging estruturado, request ID por requisição, log de latência/tokens por chamada de LLM; `/health` com semântica de readiness (503).
11. **Testes de integração:** hoje só utilitários puros são testados. Adicionar testes de API com `httpx.ASGITransport` + fakes (o desenho com DI torna isso barato) cobrindo o fluxo do chat, o fork de conversa (B1) e os erros de provedor.

## Melhorias Futuras

- **RAG real (v0.2):** pgvector (trocar a imagem do compose por `pgvector/pgvector:pg16` desde já), embeddings na escrita da memória, busca vetorial no `retrieve_relevant` — o contrato já está pronto.
- **Extração automática de memórias:** pós-processamento do turno com chamada barata (Haiku) para propor memórias; exige a sanitização do B4 antes.
- **Tool use no contrato do provider:** pré-requisito para Planner/Research/Automation Agents.
- **Sumarização de conversas longas:** quando o histórico estourar o orçamento de tokens, sumarizar as mensagens antigas em vez de cortá-las (compaction).
- **Voz (v0.3) e interface (v0.4):** streaming (item 6) é pré-requisito de ambos.
- **Título automático de conversas** (`Conversation.title` existe e nunca é preenchido).

---

## Nota Técnica

| Dimensão | Nota | Justificativa |
|---|---|---|
| Arquitetura | **7,5** | Camadas corretas, DI, abstração de provider e ponto de troca para RAG. Perde por estrutura do repo, acoplamento em `get_settings` e ausência de entidade User. |
| Código | **7,5** | Tipado, legível, imutabilidade onde importa, decisões documentadas, testes passam. Perde por código morto, validações tardias e ausência de lint/CI. |
| Segurança | **3,0** | Higiene básica correta (env, ORM, CORS, validação de entrada), mas sem autenticação, sem autorização e sem rate limiting — os três pilares ausentes. |
| IA | **6,5** | Integração e abstração corretas; memória honesta para v0.1. Sem streaming, sem orçamento de tokens, cache de prompt desperdiçado, superfície de injeção aberta. |
| Escalabilidade | **5,0** | Async de ponta a ponta, mas a conexão presa no LLM limita a ~15 requisições simultâneas e o ranking de memórias é O(n) em RAM. |
| Preparação para produção | **3,5** | Sem migrations versionadas, sem Dockerfile da app, sem CI, sem observabilidade, health check com semântica errada, onboarding quebrado (`.env.example`). |

---

## Plano de Evolução

### Yui v0.1.2 — Correções (1–2 semanas)
- Migration inicial + `alembic/versions/` versionado; `.env.example`.
- Rehidratação do Redis a partir do PostgreSQL (C4).
- Sessão de banco curta em volta do LLM + timeout explícito no SDK (C3).
- 404 para `conversation_id` inexistente (B1); coluna de sequência em `messages` (B2); validação de paridade (B3); fail-fast do provedor no boot (B6); 502 genérico (B5).
- Achatar estrutura do repositório; remover código morto.

### Yui v0.2 — Qualidade, segurança e RAG (3–6 semanas)
- Autenticação (JWT/API key) + entidade `User` + rate limiting com orçamento de tokens.
- pgvector + embeddings + busca vetorial em `retrieve_relevant`; delimitação de memórias no prompt.
- Streaming (SSE) no `/chat`.
- Tooling: pyproject + ruff + mypy + lockfile + CI + Dockerfile; testes de integração da API.
- Logging estruturado + persistência de uso de tokens; readiness no `/health`.

### Yui v0.3 — IA avançada
- Tool use no contrato `LLMProvider`; interface comum de agente.
- Memory Agent (extração automática de memórias com dedupe e decaimento) e Guardian Agent (sanitização de conteúdo que entra no prompt).
- Sumarização/compaction de conversas longas; orçamento de contexto por tokens; prompt caching estruturado.
- Voz: STT/TTS sobre o canal de streaming.

### Yui v1.0 — Produto completo
- Multiusuário completo (onboarding, isolamento testado, quotas por usuário).
- Planner/Research/Automation Agents sobre a base de tool use.
- Interface avançada + avatar consumindo streaming.
- Observabilidade completa (métricas, tracing, alertas de custo), backups, política de retenção/privacidade de dados (LGPD — o produto armazena dados pessoais sensíveis).
- Deploy reprodutível (imagens versionadas, migrations no pipeline, staging).
