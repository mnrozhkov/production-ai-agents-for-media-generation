"""Cost accounting, GPU placement policy, and thread-safe two-level Budget.

Implements the Lesson 1.5 and Module 3 accounting formulas:
* Marginal compute cost: ``c = (rate_usd_per_hour / 3600) * t_gen_s``
* Fixed worker overhead: ``F = (rate_usd_per_hour / 3600) * (startup_s + load_s + idle_s)``
* Thread-safe ``Budget`` with ``reserve / commit / release`` across both ``record_budget_usd``
  and ``run_budget_usd``.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Literal

State = Literal["ATTESTED", "DERIVED", "ESTIMATED", "UNVERIFIABLE"]
RETRIEVED = "2026-11-02"
SOURCE_PRICE_PAGE = "https://nebius.com/prices (fetched 2026-11-02)"
SOURCE_SPOT = "billing v1alpha1 spot-price list, snapshot 2026-11-02T15:20:53Z"
SOURCE_DOCS_PAGE = "https://docs.nebius.com/compute/resources/pricing (fetched 2026-11-02)"

_PRESET_GPU = re.compile(r"(\d+)gpu-", re.I)
_PRESET_CPU = re.compile(r"(\d+)vcpu-", re.I)
_PRESET_RAM = re.compile(r"-(\d+)gb", re.I)

ROLE_PLACEMENT: dict[str, tuple[str, str]] = {
    "image": ("gpu-l40s-a", "1gpu-8vcpu-32gb"),
    "speech": ("gpu-l40s-a", "1gpu-8vcpu-32gb"),
    "audio": ("gpu-l40s-a", "1gpu-8vcpu-32gb"),
    "video": ("gpu-h100-sxm", "1gpu-16vcpu-200gb"),
}


class BudgetExceeded(RuntimeError):
    """Raised when a reservation would exceed either record_budget_usd or run_budget_usd."""

    def __init__(self, message: str, *, scope: Literal["record", "run"], record_id: str) -> None:
        super().__init__(message)
        self.scope = scope
        self.record_id = record_id


@dataclass(frozen=True)
class Rate:
    """Hourly USD rate for a platform/preset, with provenance."""

    usd_per_hour: float
    on_demand_usd_per_hour: float
    basis: str
    state: State = "ATTESTED"
    source: str = SOURCE_PRICE_PAGE
    retrieved: str = RETRIEVED


def parse_preset(preset: str) -> tuple[int, int, int]:
    gpus = int(m.group(1)) if (m := _PRESET_GPU.search(preset)) else 0
    vcpus = int(m.group(1)) if (m := _PRESET_CPU.search(preset)) else 0
    ram = int(m.group(1)) if (m := _PRESET_RAM.search(preset)) else 0
    return gpus, vcpus, ram


def default_platform(role: str | None = None) -> str:
    return ROLE_PLACEMENT.get(role or "image", ROLE_PLACEMENT["image"])[0]


def default_preset(role: str | None = None) -> str:
    return ROLE_PLACEMENT.get(role or "image", ROLE_PLACEMENT["image"])[1]


def hourly_rate(platform: str, preset: str, *, preemptible: bool = False) -> Rate:
    """Compute hourly USD rate for ``platform`` and ``preset``."""
    gpus, vcpus, ram = parse_preset(preset)
    count = max(gpus, 1)
    plat = platform.lower()

    if plat.startswith("gpu-h100"):
        on_demand = round(4.50 * count, 6)
        if preemptible:
            return Rate(
                round(0.79 * count, 6),
                on_demand,
                basis="spot market price (preemptible)",
                state="ATTESTED",
                source=SOURCE_SPOT,
            )
        return Rate(on_demand, on_demand, basis="on-demand public price list")

    if plat.startswith("gpu-h200"):
        on_demand = round(5.40 * count, 6)
        if preemptible:
            return Rate(
                round(1.85 * count, 6),
                on_demand,
                basis="spot market price (preemptible)",
                state="ATTESTED",
                source=SOURCE_SPOT,
            )
        return Rate(on_demand, on_demand, basis="on-demand public price list")

    if plat.startswith("gpu-l40s"):
        cpu_rate = 0.010 if plat.startswith("gpu-l40s-d") else 0.012
        on_demand = round(1.35 * count + vcpus * cpu_rate + ram * 0.0032, 6)
        if preemptible:
            spot = round(0.65 * count + vcpus * 0.006 + ram * 0.0016, 6)
            return Rate(
                spot,
                on_demand,
                basis="ESTIMATED preemptible L40S component rates",
                state="ESTIMATED",
                source=SOURCE_DOCS_PAGE,
            )
        return Rate(
            on_demand,
            on_demand,
            basis="DERIVED: L40S components (GPU + vCPU + RAM)",
            state="DERIVED",
        )

    on_demand = round(max(vcpus, 1) * 0.033, 6)
    return Rate(on_demand, on_demand, basis="DERIVED: CPU list price", state="DERIVED")


def marginal_cost_usd(
    platform: str, preset: str, gen_seconds: float, *, preemptible: bool = False
) -> float:
    """Marginal compute cost ``c = r * t_gen`` for a single generation."""
    if gen_seconds < 0 or not math.isfinite(gen_seconds):
        raise ValueError(f"gen_seconds must be finite and nonnegative, got {gen_seconds}")
    rate = hourly_rate(platform, preset, preemptible=preemptible)
    return round(rate.usd_per_hour * (gen_seconds / 3600.0), 6)


def fixed_overhead_usd(
    platform: str,
    preset: str,
    *,
    startup_s: float | None,
    ready_or_load_s: float | None,
    idle_s: float = 0.0,
    preemptible: bool = False,
) -> float | None:
    """Fixed worker overhead ``F = r * (startup_s + t_ready + idle_s)``.

    Returns ``None`` if either ``startup_s`` or ``ready_or_load_s`` was not measured, matching
    Lesson 1.5 (never invent 0 for an unmeasured startup/load window).
    """
    if startup_s is None or ready_or_load_s is None:
        return None
    total_s = startup_s + ready_or_load_s + idle_s
    rate = hourly_rate(platform, preset, preemptible=preemptible)
    return round(rate.usd_per_hour * (total_s / 3600.0), 6)


def cost_basis(platform: str, preset: str, *, preemptible: bool = False) -> str:
    rate = hourly_rate(platform, preset, preemptible=preemptible)
    return f"{rate.basis} · {rate.state} · {rate.source} · retrieved {rate.retrieved}"


def token_cost_usd(
    tokens_in: int,
    tokens_out: int,
    *,
    price_in_per_1m_usd: float = 0.15,
    price_out_per_1m_usd: float = 0.60,
) -> float:
    return round(
        (tokens_in * price_in_per_1m_usd + tokens_out * price_out_per_1m_usd) / 1_000_000.0,
        6,
    )


class Budget:
    """Thread-safe two-level budget enforcing ``record_budget_usd`` and ``run_budget_usd``."""

    def __init__(self, *, record_budget_usd: float, run_budget_usd: float) -> None:
        if record_budget_usd <= 0 or run_budget_usd <= 0:
            raise ValueError("Budgets must be positive USD amounts")
        self.record_budget_usd = float(record_budget_usd)
        self.run_budget_usd = float(run_budget_usd)
        self._lock = threading.Lock()
        self._run_spent = 0.0
        self._run_reserved = 0.0
        self._record_spent: dict[str, float] = {}
        self._record_reserved: dict[str, float] = {}

    def reserve(self, record_id: str, estimate_usd: float) -> float:
        """Reserve ``estimate_usd`` before a paid call. Raises ``BudgetExceeded`` if over cap."""
        est = max(0.0, float(estimate_usd))
        with self._lock:
            rec_spent = self._record_spent.get(record_id, 0.0)
            rec_res = self._record_reserved.get(record_id, 0.0)
            if rec_spent + rec_res + est > self.record_budget_usd + 1e-9:
                raise BudgetExceeded(
                    f"record {record_id!r} budget exceeded: spent={rec_spent:.4f}, "
                    f"reserved={rec_res:.4f}, estimate={est:.4f}, limit={self.record_budget_usd:.4f}",
                    scope="record",
                    record_id=record_id,
                )
            if self._run_spent + self._run_reserved + est > self.run_budget_usd + 1e-9:
                raise BudgetExceeded(
                    f"run budget exceeded: spent={self._run_spent:.4f}, "
                    f"reserved={self._run_reserved:.4f}, estimate={est:.4f}, limit={self.run_budget_usd:.4f}",
                    scope="run",
                    record_id=record_id,
                )
            self._record_reserved[record_id] = rec_res + est
            self._run_reserved += est
            return est

    def commit(self, record_id: str, reserved_usd: float, actual_usd: float) -> None:
        """Replace a prior reservation with ``actual_usd`` spent."""
        res = max(0.0, float(reserved_usd))
        act = max(0.0, float(actual_usd))
        with self._lock:
            self._record_reserved[record_id] = max(
                0.0, self._record_reserved.get(record_id, 0.0) - res
            )
            self._run_reserved = max(0.0, self._run_reserved - res)
            self._record_spent[record_id] = self._record_spent.get(record_id, 0.0) + act
            self._run_spent += act

    def release(self, record_id: str, reserved_usd: float) -> None:
        """Release a reservation when a launch fails before billing."""
        res = max(0.0, float(reserved_usd))
        with self._lock:
            self._record_reserved[record_id] = max(
                0.0, self._record_reserved.get(record_id, 0.0) - res
            )
            self._run_reserved = max(0.0, self._run_reserved - res)

    def record_spent(self, record_id: str) -> float:
        with self._lock:
            return round(self._record_spent.get(record_id, 0.0), 6)

    @property
    def run_spent(self) -> float:
        with self._lock:
            return round(self._run_spent, 6)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "run_spent_usd": round(self._run_spent, 6),
                "run_reserved_usd": round(self._run_reserved, 6),
                "run_limit_usd": round(self.run_budget_usd, 6),
                "record_limit_usd": round(self.record_budget_usd, 6),
            }
