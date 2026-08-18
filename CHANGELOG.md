# Changelog

Todas as mudanças relevantes deste projeto serão documentadas neste arquivo.

## [0.5.1] - 2026-08-18

### Alterado

- Baseline oficial consolidado em Python 3.11.
- Hardening de segurança, configuração, memória e isolamento consolidado no
  repositório canônico.
- Streaming e tool calling preservados nos fluxos normal e protegido.

### Corrigido

- Falhas do cache Redis agora degradam com segurança para o PostgreSQL, que
  permanece como fonte de verdade, sem mascarar erros reais do banco (#9).
- Respostas finais vazias ou compostas apenas por whitespace são rejeitadas;
  cada turno recebe no máximo um retry global e usa fallback não vazio após o
  esgotamento (#10).

### Validado

- 262 testes automatizados, Ruff e mypy em Python 3.11.
- PostgreSQL 16 real com pgvector 0.8.6 e migrations até
  `0005_embedding_provenance`.
- Startup, autenticação, persistência, Redis indisponível, streaming, tool
  calling e inferência real com Ollama.

### Limitações conhecidas

- O GitHub Actions encerra em `startup_failure` antes de criar jobs. A v0.5.1
  foi validada pelo conjunto automatizado local e pelos smokes reais descritos
  acima; esta é uma limitação operacional do processo de CI, não uma falha
  funcional conhecida da Yui.

[0.5.1]: https://github.com/Caiobqz/Yui-Project/releases/tag/v0.5.1
