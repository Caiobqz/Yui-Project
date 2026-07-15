"""Observability — registro de métricas em processo (determinístico, sem deps).

Contadores e amostras para: decisões autônomas, memórias recuperadas,
attention score, tempo de raciocínio, uso do Goal Engine, intervenções da
Bússola Moral e confiança das decisões.

Single-process + asyncio cooperativo: incrementos são síncronos e atômicos
entre awaits, então não há corrida sem locks. Se a Yui escalar para múltiplos
processos, exportar para um backend externo (Prometheus) — os pontos de
instrumentação já estarão no lugar.
"""
from dataclasses import dataclass, field

_MAX_SAMPLES = 1000


@dataclass
class MetricsRegistry:
    counters: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[float]] = field(default_factory=dict)

    def incr(self, name: str, by: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + by

    def observe(self, name: str, value: float) -> None:
        bucket = self.samples.setdefault(name, [])
        bucket.append(float(value))
        if len(bucket) > _MAX_SAMPLES:
            del bucket[0 : len(bucket) - _MAX_SAMPLES]

    def snapshot(self) -> dict[str, object]:
        stats: dict[str, dict[str, float]] = {}
        for name, values in self.samples.items():
            if not values:
                continue
            ordered = sorted(values)
            p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
            stats[name] = {
                "count": len(values),
                "avg": round(sum(values) / len(values), 4),
                "p95": round(ordered[p95_index], 4),
                "last": round(values[-1], 4),
            }
        return {"counters": dict(self.counters), "samples": stats}

    def reset(self) -> None:
        self.counters.clear()
        self.samples.clear()


# Registro global do processo.
METRICS = MetricsRegistry()
