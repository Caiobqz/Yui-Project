"""World Model — separa explicitamente os quatro domínios de conhecimento.

    - self         o que a Yui sabe sobre si (Self Model)
    - user         o que a Yui sabe sobre o usuário (memórias, perfil, objetivos)
    - environment  o que a Yui sabe sobre o ambiente (data/hora, locale, canal)
    - general      conhecimento geral do mundo (pertence ao modelo de linguagem)

Os domínios NUNCA se misturam: cada um vira um bloco rotulado e distinto no
prompt. Este módulo cuida dos domínios `self`, `environment` e `general`
(determinísticos); o domínio `user` é montado pelos blocos de memória,
relacionamento e adaptação já existentes, sob a coordenação do orquestrador.
"""
from dataclasses import dataclass
from datetime import datetime

from app.cognition.self_model import SelfModel
from app.models.base import utcnow

_WEEKDAYS_PT = (
    "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
    "sexta-feira", "sábado", "domingo",
)


@dataclass(frozen=True)
class WorldModel:
    self_model: SelfModel
    now: datetime
    locale: str = "pt-BR"

    def environment_prompt(self) -> str:
        weekday = _WEEKDAYS_PT[self.now.weekday()]
        return (
            f"Data e hora atuais (UTC): {self.now.strftime('%Y-%m-%d %H:%M')} "
            f"({weekday}). Idioma padrão do usuário: {self.locale}. "
            "Use estes dados ao interpretar referências temporais relativas."
        )

    @staticmethod
    def general_boundary() -> str:
        return (
            "Conhecimento geral do mundo vem do seu treinamento e pode estar "
            "desatualizado — trate-o como distinto dos fatos sobre o usuário e "
            "sinalize incerteza quando a informação puder ter mudado."
        )


def build_world_model(self_model: SelfModel, now: datetime | None = None) -> WorldModel:
    return WorldModel(self_model=self_model, now=now or utcnow())
