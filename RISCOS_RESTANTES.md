# Riscos restantes

> Documento incorporado ao projeto unificado em 2026-08-14 a partir do pacote
> `fix-yui-core-hardening-v2`.

## Sobre guard_model_output — leia isto primeiro

**`guard_model_output()` é uma mitigação heurística, não uma garantia
absoluta contra prompt injection.** Ela só bloqueia a resposta quando o
TEXTO DA SAÍDA do modelo bate um dos padrões listados em
`_OUTPUT_IDENTITY_BREAK_PATTERNS` / `_OUTPUT_INTERNAL_DISCLOSURE_PATTERNS`
(ex.: "não sou Yui", "sou apenas Qwen", "meu system prompt"). Um modelo
que obedece ao ataque em substância — segue a instrução injetada, muda
de comportamento, ignora as regras — **sem usar nenhuma dessas frases
específicas**, não é pego por esta camada. O mesmo vale para
`assess_user_input()`: é um classificador por regex/palavra-chave;
paráfrase suficientemente diferente do texto de entrada, outro idioma,
ou truques de encoding podem não bater os padrões (falso negativo) —
o oposto do falso positivo que corrigimos nesta rodada.

Isso não invalida a camada — ela pega exatamente os casos concretos que
vocês relataram (incluindo o teste real com Qwen) e adiciona defesa
onde antes não havia nenhuma checagem de saída. Mas não deve ser
apresentada, internamente ou para usuários, como uma garantia
estrutural equivalente a "isto nunca pode acontecer". É a mesma
distinção do relatório anterior: separação de role (`system_prompt`
vs. `role=user`) é estrutural e vale sempre; classificação de
entrada/saída é heurística e vale até onde os padrões alcançam.

## Herdados da auditoria anterior, ainda não endereçados

- **Contradição semântica com baixa similaridade**: `reinforce()` só
  substitui o conteúdo quando `find_duplicate` já considera as duas
  memórias equivalentes (embedding ≥ `duplicate_threshold` ou Jaccard
  ≥ 0.8). Abaixo disso, duas memórias contraditórias continuam
  coexistindo como entradas separadas.
- **Label injection na transcrição de `analyzer.py`/`summary_service.py`**:
  ambos achatam o histórico em texto tipo `f"Usuário: {texto}\nYui: {resposta}"`
  antes de mandar pro modelo utilitário. Uma mensagem contendo
  `\nYui: ` ou `\nUsuário: ` literal poderia tentar forjar falas dentro
  dessa transcrição única. Não escala a privilégio (continua dentro de
  uma mensagem `role=user`), mas pode enviesar o que é extraído.
- **Tag desconhecida sobrevive dentro do bloco de memória sanitizado**:
  a sanitização em `context_service.py` remove só as 8 tags que o
  próprio sistema usa, não tags arbitrárias. Documentado com teste
  dedicado (`test_context_service.py` da auditoria anterior); mitigado
  só por instrução textual, não por estrutura.
- **Redis sem senha** no `docker-compose.yml` (baixo risco hoje — bind
  em `127.0.0.1`, não `0.0.0.0`).
- **Dockerfile só instala `requirements.txt` núcleo** — a imagem
  construída a partir dele só funciona com `LLM_PROVIDER=ollama` fora
  da caixa. Pode ser intencional.
- **Compatibilidade Windows do `llama_cpp_provider.py`** não testada
  (mais provável ser packaging/wheel do `llama-cpp-python` que código).

## Novos, desta rodada

- **`security_directive` é reforço textual no system prompt** — uma
  camada a mais de defesa em profundidade, mas ainda depende do modelo
  seguir a instrução. Para modelos locais pequenos (o cenário que
  motivou este trabalho), isso é uma mitigação parcial, não uma
  barreira. É complementar às camadas estruturais (separação de role)
  e determinísticas (`guard_model_output` no texto final), não um
  substituto delas.
- **Streaming de turnos suspeitos perde a UX incremental** — trade-off
  aceito explicitamente a pedido do usuário (bufferizar > mostrar
  conteúdo não validado), mas vale registrar que turnos marcados como
  suspeitos (inclusive falsos positivos que ainda não foram
  encontrados) vão parecer "travados" até a resposta completa chegar,
  em vez de aparecer palavra por palavra.
- **`assess_user_input` continua sendo a única fonte de verdade sobre
  "é suspeito"** — calculado uma vez por turno agora (sem duplicação),
  mas se essa classificação errar (falso negativo), nenhuma das
  camadas seguintes (directive, guard, retrieval defensivo, bloqueio
  de pós-turno) é acionada, porque todas dependem do mesmo
  `SecurityAssessment`. É uma escolha de design correta (elimina
  inconsistência entre camadas), mas concentra toda a defesa
  comportamental num único ponto de decisão inicial.
