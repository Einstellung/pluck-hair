"""Default detection task (no tracker yet)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.core.vision.pipeline import VisionPipeline
from src.scheduler.tasks.base import Task, TaskIterationResult
from src.scheduler.tasks.conditions import DoneCondition, AlwaysFalseCondition
from src.scheduler.tasks.stats import TaskStats


class DetectionTask(Task):
    """Wrap VisionPipeline into a Task without tracking."""

    def __init__(
        self,
        pipeline: VisionPipeline,
        *,
        done_condition: Optional[DoneCondition] = None,
        stats: Optional[TaskStats] = None,
        name: str = "detection",
    ):
        super().__init__(
            pipeline,
            done_condition=done_condition or AlwaysFalseCondition(),
            stats=stats,
        )
        self._name = name
        self._frame_count = 0

    @property
    def name(self) -> str:
        return self._name

    def run_iteration(self, image: np.ndarray) -> TaskIterationResult:
        pipeline_result = self.pipeline.run(image)
        self._frame_count += 1

        result = TaskIterationResult(
            detections=pipeline_result.detections,
            is_done=False,
            metadata={
                "pipeline_time_ms": pipeline_result.processing_time_ms,
                "detection_count": len(pipeline_result.detections),
                "frame_in_task": self._frame_count,
            },
        )

        self.stats.record(result)
        result.is_done = self.done_condition.check(self.stats, result)
        return result

    def reset(self) -> None:
        self._frame_count = 0
        self.stats.reset()
        self.done_condition.reset()
