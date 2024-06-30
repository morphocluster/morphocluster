import time
from morphocluster.server import jobs
from rq.job import JobStatus


def test_worker(flask_rq_worker):

    job = jobs.add.queue(1, 2)

    # Wait for job to complete
    for i in range(10):
        status = job.get_status()

        assert status != JobStatus.FAILED

        if status == JobStatus.FINISHED:
            break

        time.sleep(1.0)
