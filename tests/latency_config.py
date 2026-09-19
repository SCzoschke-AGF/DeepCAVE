"""Isolated configuration for repeatable local browser-latency measurements."""

import os
from pathlib import Path

from deepcave.config import Config as BaseConfig


class Config(BaseConfig):
    """Keep latency-probe state outside the repository and user metacache."""

    DASH_PORT = int(os.environ.get("DEEPCAVE_LATENCY_PORT", "8061"))
    REDIS_PORT = int(os.environ.get("DEEPCAVE_LATENCY_REDIS_PORT", "6381"))
    META_DEFAULT = {
        "working_dir": str(
            Path(os.environ.get("DEEPCAVE_LATENCY_RUN", "/tmp/deepcave-latency/run")).parent
        ),
        "selected_run_paths": [os.environ.get("DEEPCAVE_LATENCY_RUN", "/tmp/deepcave-latency/run")],
        "groups": {},
        "groups_configspace_congruence": {},
        "active_run_id": None,
    }

    @property
    def CACHE_DIR(self) -> Path:
        """Use a disposable cache without touching normal DeepCAVE state."""
        return Path(os.environ.get("DEEPCAVE_LATENCY_CACHE", "/tmp/deepcave-latency/cache"))
