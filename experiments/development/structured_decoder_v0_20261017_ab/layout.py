"""Audit A1's fixed-stride grid without silently dropping the record tail."""
from __future__ import annotations


def strict_grid_layout(record_samples):
    if record_samples < 1600:
        raise ValueError("Record does not supply one full 8-second input")
    starts = list(range(0, record_samples-1600+1, 1024))
    retained = [(start+288, start+1312) for start in starts]
    assert all(a[1] == b[0] for a,b in zip(retained,retained[1:]))
    requested_stop = record_samples-288
    return {"record_samples": record_samples, "input_starts": starts,
            "retained": retained, "requested_support": [288, requested_stop],
            "actual_support": [288, retained[-1][1]],
            "uncovered_tail_samples": requested_stop-retained[-1][1],
            "complete_coverage": requested_stop == retained[-1][1]}


def proposed_tail_window(record_samples):
    """Geometry only, not authorization to use a non-stride-aligned input."""
    plan = strict_grid_layout(record_samples)
    if plan["complete_coverage"]:
        return None
    return {"additional_input": [record_samples-1600, record_samples],
            "central_output": [record_samples-1312, record_samples-288],
            "retain_only": [plan["actual_support"][1], record_samples-288]}


def approved_layout(record_samples):
    """Author-approved 2026-10-17 end-aligned exception, half-open ownership."""
    plan = strict_grid_layout(record_samples)
    segments = [dict(input_start=s, start=s+288, stop=s+1312, tail=False)
                for s in plan["input_starts"]]
    tail = proposed_tail_window(record_samples)
    if tail is not None:
        segments.append(dict(input_start=tail["additional_input"][0],
            start=tail["retain_only"][0], stop=tail["retain_only"][1], tail=True))
    assert segments[0]["start"] == 288
    assert segments[-1]["stop"] == record_samples-288
    assert all(a["stop"] == b["start"] for a,b in zip(segments,segments[1:]))
    assert sum(s["stop"]-s["start"] for s in segments) == record_samples-576
    return segments


def dense_starts(record_samples, stride=200):
    if stride != 200:
        raise ValueError("Dense allocation stride is frozen at 200 samples")
    return sorted(set(range(0, record_samples-1599, stride)) | {record_samples-1600})
