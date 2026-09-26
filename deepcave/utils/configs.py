# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  noqa: D400
"""
# Configs

This module provides a utility for parsing the configurations from a filename.
"""

from typing import Optional

import importlib.util
from pathlib import Path

from deepcave.config import Config


def parse_config(filename: Optional[str] = None) -> Config:
    """
    Parse the config given the filename.

    Both relative and absolute paths are possible.

    Parameters
    ----------
    filename : Optional[str], optional
        Location of the config. Must be a python file.
        By default None (default configuration will be used).

    Note
    ----
    The python file must contain a class named ``Config`` and inherit ``deepcave.config.Config``.

    Returns
    -------
    Config
        Either the default config (if no filename is given) or the config parsed from the given
        filename.

    Raises
    ------
    RuntimeError
        If config class could not be loaded.
    """
    config = Config()
    if filename is not None and filename != "default":
        try:
            p = Path(filename).expanduser().resolve()
            spec = importlib.util.spec_from_file_location("deepcave_user_config", p)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create an import spec for {p}.")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config = module.Config()

        except Exception as exc:
            raise RuntimeError(f"Could not load class Config from {p}.") from exc

    return config
