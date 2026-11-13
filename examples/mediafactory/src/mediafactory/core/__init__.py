"""Core contracts, cost math, trace events, and FUSE-safe storage (zero GPU/cloud SDK imports)."""

from mediafactory.core.cost import (
    Budget,
    BudgetExceeded,
    Rate,
    cost_basis,
    default_platform,
    default_preset,
    fixed_overhead_usd,
    hourly_rate,
    marginal_cost_usd,
)
from mediafactory.core.events import EventLogger, TraceEvent
from mediafactory.core.result import (
    AssetMeta,
    CheckReport,
    CostReport,
    DeterministicCheck,
    Measurements,
    ModelRef,
    ResultError,
    ResultReport,
    SpeechCheck,
)
from mediafactory.core.spec import (
    GenerationSpec,
    InputImage,
    LoraAdapter,
    Pricing,
    Role,
    Size,
    SpecError,
)
from mediafactory.core.storage import DiskStorage, MemoryStorage, StorageError, staged_write

__all__ = [
    "AssetMeta",
    "Budget",
    "BudgetExceeded",
    "CheckReport",
    "CostReport",
    "DeterministicCheck",
    "DiskStorage",
    "EventLogger",
    "GenerationSpec",
    "InputImage",
    "LoraAdapter",
    "Measurements",
    "MemoryStorage",
    "ModelRef",
    "Pricing",
    "Rate",
    "ResultError",
    "ResultReport",
    "Role",
    "Size",
    "SpecError",
    "SpeechCheck",
    "StorageError",
    "TraceEvent",
    "cost_basis",
    "default_platform",
    "default_preset",
    "fixed_overhead_usd",
    "hourly_rate",
    "marginal_cost_usd",
    "staged_write",
]
