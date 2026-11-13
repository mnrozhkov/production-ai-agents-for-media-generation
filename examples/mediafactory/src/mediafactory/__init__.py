"""Personalized Media Factory — general-purpose multimodal agent and batch engine."""

from mediafactory.config import PACKAGE_ROOT, Settings
from mediafactory.core.cost import Budget, BudgetExceeded, hourly_rate, marginal_cost_usd
from mediafactory.core.events import EventLogger, TraceEvent
from mediafactory.core.result import ResultReport
from mediafactory.core.spec import GenerationSpec, InputImage, LoraAdapter
from mediafactory.profile.application import ApplicationBundle, JudgeVerdict, load_application
from mediafactory.profile.records import Brief, build_brief, load_records
from mediafactory.worker.runner import run_batch_specs, run_single
from mediafactory.workflows.batch import run_batch
from mediafactory.workflows.finetune import LoraConfig, prepare_dataset, start_finetune
from mediafactory.workflows.pipeline import run_record
from mediafactory.workflows.tickets import Ticket, reconcile_or_submit

__version__ = "0.2.0"

__all__ = [
    "PACKAGE_ROOT",
    "ApplicationBundle",
    "Brief",
    "Budget",
    "BudgetExceeded",
    "EventLogger",
    "GenerationSpec",
    "InputImage",
    "JudgeVerdict",
    "LoraAdapter",
    "LoraConfig",
    "ResultReport",
    "Settings",
    "Ticket",
    "TraceEvent",
    "__version__",
    "build_brief",
    "hourly_rate",
    "load_application",
    "load_records",
    "marginal_cost_usd",
    "prepare_dataset",
    "reconcile_or_submit",
    "run_batch",
    "run_batch_specs",
    "run_record",
    "run_single",
    "start_finetune",
]
