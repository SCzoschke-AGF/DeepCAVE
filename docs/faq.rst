Frequently Asked Questions
==========================


DeepCAVE is not starting. What's wrong?
    Run ``deepcave --check_redis`` to verify the configured Redis connection. For a source
    checkout, start the included service with ``docker compose up -d redis``.

My machine is not the fastest and I have some issues (e.g. content is not updating) when using plugins with the queue. What can I do?
    It might be the case that your machine is too slow to update the content before one calculation
    circle was finished. Increasing the refresh rate could help. To do so, set `REFRESH_RATE`
    to a higher value in the DeepCAVE config file.
