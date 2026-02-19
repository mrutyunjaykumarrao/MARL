"""
Training Module
===============

Complete training infrastructure for MARL Jammer System.

Components:
    - config: Training hyperparameter configuration
    - metrics: Logging and metrics tracking
    - trainer: Main training loop

Example:
    >>> from training import Trainer, TrainingConfig
    >>> config = TrainingConfig()
    >>> trainer = Trainer(config)
    >>> trainer.train()
"""

from .config import (
    TrainingConfig,
    EnvironmentConfig,
    NetworkConfig,
    PPOConfig,
    get_debug_config,
    get_fast_config,
    get_full_config,
    get_large_scale_config
)

from .metrics import (
    MetricsLogger,
    RollingStats,
    EvaluationResult
)

from .trainer import Trainer

__all__ = [
    # Config
    "TrainingConfig",
    "EnvironmentConfig", 
    "NetworkConfig",
    "PPOConfig",
    "get_debug_config",
    "get_fast_config",
    "get_full_config",
    "get_large_scale_config",
    # Metrics
    "MetricsLogger",
    "RollingStats",
    "EvaluationResult",
    # Trainer
    "Trainer"
]
