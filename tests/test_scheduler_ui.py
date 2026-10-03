import logging
import threading

from deal_agent_framework import DealAgentFramework


class CountingFramework(DealAgentFramework):
    """The scheduler and scan_now, without agents: run() just counts."""

    def __init__(self):
        self.settings = {"scan": {"interval_seconds": 0.05}}
        self._run_lock = threading.Lock()
        self._scheduler_thread = None
        self._stop = threading.Event()
        self.runs = 0
        self.ran = threading.Event()

    def run(self, extra=None, live=False):
        with self._run_lock:
            self.runs += 1
            if self.runs >= 3:
                self.ran.set()
        return []


def test_scheduler_runs_without_any_browser_and_only_once():
    fw = CountingFramework()
    assert fw.start_scheduler() is True
    assert fw.start_scheduler() is False, "a second call must not start a second loop"
    assert fw.ran.wait(5), "scans run on their own, every interval"
    fw.stop()


def test_scan_now_skips_when_a_scan_is_running():
    fw = CountingFramework()
    with fw._run_lock:
        assert fw.scan_now() is False
    assert fw.scan_now() is True


def test_log_buffer_keeps_recent_lines_from_any_thread():
    from price_is_right import LogBuffer

    buffer = LogBuffer(size=3)
    logger = logging.getLogger("buffer-test")
    logger.addHandler(buffer)
    logger.setLevel(logging.INFO)
    worker = threading.Thread(target=lambda: [logger.info(f"line {i}") for i in range(5)])
    worker.start()
    worker.join()
    logger.removeHandler(buffer)
    assert [line.split("] ")[-1] for line in buffer.lines] == ["line 2", "line 3", "line 4"]
    assert "line 4" in buffer.html()
