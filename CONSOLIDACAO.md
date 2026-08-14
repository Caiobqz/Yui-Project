# Consolidação do projeto Yui

Data da consolidação: **2026-08-14**.

## Proveniência

A versão unificada foi montada nesta ordem:

1. `main` de `Caiobqz/Yui-Project`, no commit
   `c279433b6a034739f047d5b5af99e13bb82f1f7c`;
2. alterações adicionais da versão `Yui-Project-AT` em
   `app/agents/guardian.py`, `app/agents/yui_core.py` e
   `app/cognition/identity.py`;
3. os sete patches, na ordem original, do pacote
   `fix-yui-core-hardening-v2`.

O snapshot `Yui-c279433` foi comparado por hash com o checkout do GitHub e
correspondia integralmente ao commit informado acima. Depois de normalizar
quebras de linha, a versão AT diferia semanticamente apenas nos três arquivos
listados.

## Estrutura canônica

Existe uma única implementação executável, na raiz do projeto:

- `app/` — aplicação;
- `tests/` — suíte de testes;
- `alembic/` — migrações;
- arquivos de configuração e dependências na raiz.

A árvore legada `yui-ai-v0.1.1/` foi removida. Seus 138 arquivos rastreados
tinham equivalentes na implementação da raiz e nenhum arquivo funcional
exclusivo. O arquivo aninhado `Yui-Project-v0.4.zip` também foi removido para
evitar distribuir outra cópia do próprio projeto.

Também foram removidos 91 artefatos Python compilados (`__pycache__`/`.pyc`)
que estavam indevidamente rastreados no histórico. Esses arquivos já são
cobertos pelo `.gitignore` e devem ser recriados localmente quando necessário.

## Revisão adicional

Durante a validação em Windows, foi corrigido um teste que comparava um caminho
usando separadores Unix. A asserção agora usa os componentes de `pathlib.Path`,
mantendo o mesmo comportamento em Windows e Linux.

## Validação da versão unificada

Ambiente: Python 3.11.9.

- `pytest -q`: **245 passed**;
- `ruff check app tests alembic`: **sem erros**;
- `mypy app`: **sem erros em 93 arquivos**;
- `alembic upgrade head` com SQLite temporário: **migrações 0001 a 0005
  aplicadas com sucesso**.

As limitações de segurança que ainda exigem atenção estão registradas em
[`RISCOS_RESTANTES.md`](RISCOS_RESTANTES.md).
