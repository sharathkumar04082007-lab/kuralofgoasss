import time
from typing import Dict, Optional
from contextlib import contextmanager
from pipeline.schemas import LatencyBreakdown


class LatencyTracker:
    """
    High-resolution monotonic timer for tracking microsecond and millisecond stage durations.
    """

    def __init__(self):
        self.stage_times: Dict[str, float] = {
            "stt_ms": 0.0,
            "guardrails_ms": 0.0,
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "generation_ms": 0.0,
            "grounding_ms": 0.0,
            "total_ms": 0.0
        }
        self._start_perf = time.perf_counter()

    @contextmanager
    def track(self, stage_name: str):
        """Context manager to measure execution time of a specific pipeline stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            key = f"{stage_name}_ms" if not stage_name.endswith("_ms") else stage_name
            self.stage_times[key] = round(self.stage_times.get(key, 0.0) + elapsed_ms, 2)

    def record_stage(self, stage_name: str, elapsed_ms: float) -> None:
        """Manually record a stage duration."""
        key = f"{stage_name}_ms" if not stage_name.endswith("_ms") else stage_name
        self.stage_times[key] = round(elapsed_ms, 2)

    def get_breakdown(self) -> LatencyBreakdown:
        """Calculate total and return populated LatencyBreakdown model."""
        total = (time.perf_counter() - self._start_perf) * 1000.0
        self.stage_times["total_ms"] = round(total, 2)
        return LatencyBreakdown(**self.stage_times)
