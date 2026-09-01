from unittest.mock import Mock, patch

from redis.exceptions import ConnectionError

from deepcave.utils.redis import create_redis_connection, is_redis_available


@patch("deepcave.utils.redis.redis.from_url")
def test_create_redis_connection(from_url: Mock) -> None:
    create_redis_connection("redis://localhost", 6379)

    from_url.assert_called_once_with("redis://localhost:6379", socket_timeout=None)


@patch("deepcave.utils.redis.redis.from_url")
def test_redis_available(from_url: Mock) -> None:
    connection = from_url.return_value
    connection.ping.return_value = True

    assert is_redis_available("redis://localhost", 6379)
    from_url.assert_called_once_with(
        "redis://localhost:6379",
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
    )


@patch("deepcave.utils.redis.redis.from_url")
def test_redis_unavailable(from_url: Mock) -> None:
    from_url.return_value.ping.side_effect = ConnectionError

    assert not is_redis_available("redis://localhost", 6379)
