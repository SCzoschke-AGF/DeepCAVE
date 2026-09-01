Run Redis
=========

DeepCAVE requires a Redis server for its background job queue. At startup it sends ``PING`` to
the configured ``REDIS_ADDRESS`` and ``REDIS_PORT``. The server can run in Docker, on the host, or
on another trusted system.

Docker Compose (recommended)
----------------------------

The repository includes a Redis service for local development:

.. code:: bash

    docker compose up -d redis
    docker compose ps redis

The service publishes port 6379 only on the host's loopback interface. RDB and AOF persistence
are disabled because DeepCAVE queue jobs are temporary and should not survive Redis restarts.

To stop the service, run:

.. code:: bash

    docker compose down

No local ``redis-server`` or ``redis-cli`` binary is required. To test the same connection path
that DeepCAVE uses, activate its Python environment and run:

.. code:: bash

    deepcave --check_redis

System-wide installation
------------------------

A local Redis package remains supported:

.. code:: bash

    sudo apt-get install redis-server  # Linux
    brew install redis                 # macOS

If Redis is already running, DeepCAVE connects to it. Otherwise, the launcher starts the local
``redis-server`` binary for the lifetime of DeepCAVE.

Custom server
-------------

Set ``REDIS_ADDRESS`` and ``REDIS_PORT`` in a custom DeepCAVE configuration. Keep Redis on a
trusted network and configure authentication when it is not limited to the local machine.
