"""Task abstractions for scheduler."""

from .base import Task, TaskIterationResult
from .conditions import DoneCondition, AlwaysFalseCondition
from .stats import TaskStats
from .detection import DetectionTask

__all__ = [
    "Task",
    "TaskIterationResult",
    "DoneCondition",
    "AlwaysFalseCondition",
    "TaskStats",
    "DetectionTask",
]
