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

from unittest.mock import Mock

from rq.exceptions import NoSuchJobError

from deepcave.custom_queue import Queue


def test_jobs_for_run_are_filtered_and_inactive_jobs_are_deleted() -> None:
    queue = Queue.__new__(Queue)
    running = Mock(id="running", meta={"run_id": "target"})
    other = Mock(id="other", meta={"run_id": "other"})
    finished = Mock(id="finished", meta={"run_id": "target"})
    failed = Mock(id="failed", meta={"run_id": "target"})
    jobs = {
        "running": [running, other],
        "pending": [],
        "finished": [finished],
        "failed": [failed],
    }
    queue.get_jobs = Mock(side_effect=lambda registry: jobs[registry])
    queue.delete_job = Mock()

    assert queue.has_active_jobs_for_run("target")
    queue.delete_inactive_jobs_for_run("target")

    assert [call.args[0] for call in queue.delete_job.call_args_list] == ["finished", "failed"]


def test_retry_clears_stale_failure_before_enqueuing() -> None:
    queue = Queue.__new__(Queue)
    queue._queue = Mock()
    queue.is_processed = Mock(return_value=False)
    failed = {"retry"}
    queue._queue.failed_job_registry.get_job_ids.side_effect = lambda: list(failed)
    queue._queue.failed_job_registry.remove.side_effect = failed.discard

    def enqueue(*_args, **_kwargs):
        assert not queue.has_failed("retry")

    queue._queue.enqueue.side_effect = enqueue
    assert queue.has_failed("retry")
    queue.enqueue(Mock(), [], "retry", {})
    assert not queue.has_failed("retry")
    assert queue._queue.enqueue.call_count == 1


def test_delete_removes_orphaned_registry_marker_without_fetching_job() -> None:
    queue = Queue.__new__(Queue)
    queue._queue = Mock()
    queue.get_workers = Mock(return_value=[])
    queue.get_job = Mock(side_effect=NoSuchJobError)
    failed = {"orphan"}

    def remove(job_id, delete_job):
        failed.discard(job_id)
        raise NoSuchJobError

    queue._queue.failed_job_registry.remove.side_effect = remove
    queue.delete_job("orphan")
    assert not failed
    queue.get_job.assert_not_called()
