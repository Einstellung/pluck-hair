"""Abstract base class for all data modules."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pytorch_lightning import LightningDataModule

if TYPE_CHECKING:
    from typing import Any


class BaseDataModule(LightningDataModule, ABC):
    """
    Abstract base class for all dataset data modules.

    All concrete data modules (VOC, Oxford Pets, etc.) should inherit from this class
    and implement the from_config class method.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Any) -> BaseDataModule:
        """
        Factory method to create a datamodule instance from configuration.

        Args:
            config: The experiment configuration object (typically ExperimentConfig)

        Returns:
            An instance of the concrete datamodule class

        Note:
            Each concrete implementation is responsible for extracting the necessary
            parameters from the config object and passing them to __init__.
        """
        raise NotImplementedError("Subclasses must implement from_config()")
