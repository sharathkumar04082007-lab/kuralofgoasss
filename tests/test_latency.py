import time
import pytest
from analytics.latency_tracker import LatencyTracker
from analytics.metrics_collector import MetricsCollector
from pipeline.schemas import QueryResponse, LatencyBreakdown


def test_latency_tracker_timing():
    tracker = LatencyTracker()
    with tracker.track("retrieval"):
        time.sleep(0.01) # 10ms sleep
    with tracker.track("generation"):
        time.sleep(0.005) # 5ms sleep

    breakdown = tracker.get_breakdown()
    assert breakdown.retrieval_ms >= 8.0
    assert breakdown.generation_ms >= 4.0
    assert breakdown.total_ms >= 14.0


def test_metrics_collector_percentiles():
    collector = MetricsCollector()
    latencies = [50.0, 60.0, 70.0, 80.0, 100.0, 150.0, 200.0]
    for lat in latencies:
        resp = QueryResponse(
            transcript="test",
            answer="test answer",
            sources=[],
            confidence=0.8,
            grounded=True,
            latency_ms=LatencyBreakdown(
                retrieval_ms=lat * 0.4,
                generation_ms=lat * 0.6,
                total_ms=lat
            )
        )
        collector.record(resp)

    summary = collector.generate_summary()
    total_percentiles = summary["latency_breakdown"]["total"]
    assert total_percentiles["p50"] > 0
    assert total_percentiles["p70"] > 0
    assert total_percentiles["p100"] == 200.0
    assert summary["grounded_ratio"] == 1.0
