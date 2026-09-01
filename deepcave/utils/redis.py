# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for connecting to Redis."""

from typing import Optional

import redis
from redis.exceptions import RedisError


def create_redis_connection(
    address: str, port: int, timeout: Optional[float] = None
) -> redis.Redis:
    """Create a Redis connection from DeepCAVE's address and port configuration."""
    # RQ uses a blocking Redis read while it waits for jobs. redis-py 8 defaults to a
    # five-second socket timeout, which makes RQ 1.10 stop an idle worker silently.
    options = {"socket_timeout": None}
    if timeout is not None:
        options = {
            "socket_connect_timeout": timeout,
            "socket_timeout": timeout,
        }

    return redis.from_url(f"{address}:{port}", **options)


def is_redis_available(address: str, port: int, timeout: float = 1.0) -> bool:
    """Return whether the configured Redis server responds to ``PING``."""
    try:
        connection = create_redis_connection(address, port, timeout)
        return bool(connection.ping())
    except (RedisError, ValueError):
        return False
