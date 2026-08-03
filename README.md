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
docker compose up -d               # Postgres (pgvector) + Redis
python -m venv .venv && source .venv/bin/activate

# Escolha o(s) extra(s) de acordo com o(s) backend(s) que for usar — ver
# "Inferência e Embeddings Locais" abaixo para todas as combinações.
pip install -r requirements-dev.txt   # ambiente de desenvolvimento completo

cp .env.example .env                  # preencha o provedor de IA escolhido
alembic upgrade head
uvicorn app.main:app --reload
```

A Yui roda com quatro backends de LLM (`LLM_PROVIDER`): `claude`, `openai`,
`llama_cpp` e `ollama` — os dois últimos 100% locais, sem API Key e sem
acesso à internet no uso normal. Veja a seção **Inferência e Embeddings
Locais** para configuração completa.

## Qualidade

```bash
pytest                              # 229 testes (suíte padrão: sem GGUF,
                                     # sem Ollama, sem GPU, sem internet)
ruff check app tests alembic
mypy app
```

Testes marcados `integration`, `requires_gguf` ou `requires_ollama` não
rodam por padrão — exigem um modelo real ou uma instância do Ollama em
execução. Rode manualmente com `pytest -m requires_ollama`, por exemplo,
quando tiver o ambiente disponível.

## Inferência e Embeddings Locais

A camada de LLM (`app/services/llm/`) e de embeddings
(`app/services/embeddings/`) é uma abstração por factory — o Companion
Core (Guardian, Affect, Initiative, Goal Engine, Context Orchestrator
etc.) nunca sabe qual backend está ativo. Trocar de backend é só
configuração; nenhum sistema cognitivo muda.

```
Yui Core
    │
    ▼
LLMProvider (contrato neutro: ChatMessage, ToolSpec, ToolCall,
             LLMResponse, StreamChunk, LLMCapabilities)
    │
    ├── ClaudeProvider     (comercial, requer ANTHROPIC_API_KEY)
    ├── OpenAIProvider     (comercial, requer OPENAI_API_KEY)
    ├── LlamaCppProvider   (100% local, arquivo .gguf)
    └── OllamaProvider     (100% local, servidor HTTP)
```

### Instalação por backend

O núcleo (`requirements.txt`) não inclui nenhum SDK comercial nem
biblioteca de inferência local — cada backend é um extra opcional:

```bash
pip install -r requirements-anthropic.txt         # LLM_PROVIDER=claude
pip install -r requirements-openai.txt            # LLM_PROVIDER=openai (LLM e/ou embeddings)
pip install -r requirements-local-llama.txt       # LLM_PROVIDER=llama_cpp
pip install -r requirements-local-embeddings.txt  # EMBEDDING_PROVIDER=sentence_transformers
```

Cada arquivo já inclui `-r requirements.txt` — não é preciso combinar.
`requirements-dev.txt` inclui os extras comerciais (para a suíte de testes
exercitar os quatro backends) mas **não** inclui `llama-cpp-python` nem
`sentence-transformers`: são pesadas (compilação nativa / download de
modelo) e os testes as substituem por mocks do módulo — ver
`tests/test_llama_cpp_provider.py` e
`tests/test_sentence_transformers_provider.py`.

### llama.cpp

Requer `build-essential` e `cmake` no sistema (a instalação compila
componentes nativos):

```bash
sudo apt install build-essential cmake      # Debian/Ubuntu
pip install -r requirements-local-llama.txt
```

Para offload em GPU (CUDA/Metal/ROCm), defina `CMAKE_ARGS` antes de
instalar — consulte a documentação do pacote `llama-cpp-python` para as
flags do seu runtime.

Configuração mínima (`.env`):

```env
LLM_PROVIDER=llama_cpp
LLM_MODEL_PATH=models/qwen3-8b-instruct.Q4_K_M.gguf
LLM_CONTEXT_SIZE=8192
LLM_GPU_LAYERS=0
LLM_THREADS=4
```

**Modelos** (exemplos, não fixos no código — qualquer GGUF compatível
funciona): Qwen3, Llama 3.x, Mistral, Phi. Baixe manualmente (a Yui nunca
baixa modelos automaticamente) de uma fonte como Hugging Face, e **verifique
a licença do modelo escolhido** antes de usar. RAM aproximada: um modelo
quantizado Q4_K_M de 8B ocupa ~5–6 GB; ajuste conforme o tamanho e a
quantização escolhidos.

**Concorrência:** esta versão trava a inferência em **1 requisição
simultânea por processo**, sempre — `LLM_MAX_CONCURRENT_REQUESTS` acima de
1 é ignorado (com aviso no log). Não há isolamento de contexto por
requisição (múltiplas instâncias `Llama()` independentes ou contexts
separados); uma implementação futura com essa garantia poderia elevar o
limite com segurança.

**Tool calling:** depende do `LLM_CHAT_FORMAT` configurado. Só formatos
com sufixo `-function-calling` (ex.: `chatml-function-calling`) ou
`functionary`/`functionary-v2` são reconhecidos como capazes — **`chatml`
simples nunca habilita tool calling**, mesmo que o modelo em teoria
suporte function calling por outro mecanismo. Se a aplicação tentar enviar
ferramentas a um modelo sem essa capacidade declarada, a chamada falha
com um erro claro (nunca degrada para parsing informal de texto).

### Ollama

Requer uma instância do Ollama instalada e em execução:

```bash
ollama serve                       # em outro terminal/processo
ollama pull qwen3:8b                # baixe o modelo antes de usar
```

Configuração mínima (`.env`):

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

**Tool calling:** o default é **sempre `false`**, para qualquer modelo —
não há forma confiável de detectar automaticamente se um modelo Ollama
suporta tool calling (não é exposto de forma estruturada e uniforme entre
versões). Defina `OLLAMA_SUPPORTS_TOOL_CALLING=true` somente depois de
validar manualmente que o modelo escolhido suporta.

**Streaming:** o Ollama pode não fragmentar `tool_calls` incrementalmente
— em algumas versões só aparecem completos no chunk final (`done: true`).
O `OllamaProvider` já trata isso (acumula o que vier, sem presumir
fragmentação), mas isso significa que uma ferramenta só é conhecida no
fim do stream, não progressivamente.

### Provider principal e utilitário

`UTILITY_LLM_PROVIDER`/`UTILITY_LLM_MODEL_PATH`/`OLLAMA_UTILITY_MODEL`
(vazios por padrão) permitem um backend/modelo diferente para o trabalho
cognitivo de bastidor (extração de memórias, adaptação, resumos). Quando
não configurados, o utilitário **herda o mesmo backend e modelo/arquivo do
principal** — e, para llama.cpp e Ollama, a mesma instância é reutilizada
automaticamente (a factory detecta a configuração idêntica), evitando
carregar o mesmo modelo pesado duas vezes na RAM/VRAM. É possível também
misturar backends (ex.: principal `llama_cpp` local pesado + utilitário
`ollama` local mais leve) sem exigir configuração cruzada.

### Embeddings locais (sentence-transformers)

```env
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu
```

O primeiro uso baixa os pesos do Hugging Face Hub; usos seguintes
funcionam offline. `EMBEDDING_DEVICE=cuda`/`mps` sem o hardware/instalação
correspondente falha com um erro claro — **nunca** cai para CPU
silenciosamente.

**Dimensão e proveniência:** cada vetor é validado contra a dimensão
declarada pelo provedor antes de ser persistido ou comparado — um
mismatch (ex.: trocar de modelo sem reindexar) nunca crasha um turno; a
memória afetada é ignorada apenas naquela consulta de ranking semântico
(permanece no banco, recuperável por busca lexical) e um aviso é
registrado no log. Cada memória com embedding grava também
`embedding_provider`/`embedding_model` (colunas nulas para memórias sem
vetor ou anteriores à migration `0005_embedding_provenance`) — útil para
diagnosticar quais memórias precisam de reindexação após uma troca de
modelo.

**Reindexação:** trocar de `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL` não
invalida memórias existentes automaticamente (o texto original é
preservado) — mas também não há, nesta versão, um comando de reindexação
em lote. Memórias antigas continuam funcionando via busca lexical; a
reindexação para busca semântica é um trabalho futuro.

### Múltiplos workers

Cada worker do uvicorn/gunicorn carrega sua própria cópia do modelo — N
workers = N cópias na RAM/VRAM. Para llama.cpp/Ollama locais, prefira um
único worker por instância de modelo, a menos que a máquina tenha memória
de sobra.

### Docker

O `Dockerfile` atual (`python:3.12-slim`) não inclui toolchain de
compilação — para empacotar `llama-cpp-python` em produção, adicione
`build-essential`/`cmake` (e, para GPU, a imagem base CUDA correspondente)
à etapa de build.

### Troubleshooting

| Sintoma | Causa provável |
|---|---|
| Boot falha citando `LLM_MODEL_PATH` | Backend `llama_cpp` sem caminho configurado, ou arquivo inexistente no caminho dado |
| Boot falha citando `OLLAMA_BASE_URL`/`OLLAMA_MODEL` | Backend `ollama` selecionado sem configuração mínima |
| Erro "não foi possível conectar ao Ollama" | Servidor não está rodando (`ollama serve`) ou URL incorreta |
| Erro "modelo não encontrado" (Ollama) | Modelo não baixado — rode `ollama pull <modelo>` |
| Erro "não declara suporte a tool calling" | `LLM_CHAT_FORMAT`/`OLLAMA_SUPPORTS_TOOL_CALLING` não habilita ferramentas para o modelo configurado — confirme suporte real antes de sobrescrever |
| Aviso "forçando 1" nos logs do llama.cpp | `LLM_MAX_CONCURRENT_REQUESTS` configurado acima de 1 — ignorado por design nesta versão |
| Falha ao instalar `llama-cpp-python` | Faltam `build-essential`/`cmake` no sistema |

### Criando um novo provider

Implemente `LLMProvider` (ou `EmbeddingProvider`) em
`app/services/llm/` (ou `embeddings/`), registre-o na factory
correspondente, e siga os contratos neutros já existentes
(`ChatMessage`, `ToolSpec`, `LLMResponse`, `StreamChunk`,
`LLMCapabilities`) — nenhum outro módulo deve importar SDKs ou clientes
específicos de backend.

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
