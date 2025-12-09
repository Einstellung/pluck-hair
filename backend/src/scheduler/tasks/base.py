"""Task base classes and result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.vision.pipeline import VisionPipeline
from src.core.vision.types import Detection
from .conditions import AlwaysFalseCondition, DoneCondition
from .stats import TaskStats


@dataclass
class TaskIterationResult:
    """Single iteration output from a Task."""

    detections: List[Detection]
    is_done: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """Stateful task abstraction."""

    def __init__(
        self,
        pipeline: VisionPipeline,
        *,
        done_condition: Optional[DoneCondition] = None,
        stats: Optional[TaskStats] = None,
    ):
        self.pipeline = pipeline
        self.done_condition = done_condition or AlwaysFalseCondition()
        self.stats = stats or TaskStats()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-friendly task name for logging and metrics."""
        raise NotImplementedError

    @abstractmethod
    def run_iteration(self, image: np.ndarray) -> TaskIterationResult:
        """Process a single frame."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset task state when starting a new session/region."""
        raise NotImplementedError

    def get_stats_summary(self) -> Dict[str, Any]:
        """Return a summary of task statistics."""
        return self.stats.summary()
