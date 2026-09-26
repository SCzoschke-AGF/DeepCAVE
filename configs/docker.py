"""Configuration for the Docker Compose deployment."""

from deepcave.config import Config as BaseConfig


class Config(BaseConfig):
    DASH_ADDRESS = "0.0.0.0"
    REDIS_ADDRESS = "redis://redis"
