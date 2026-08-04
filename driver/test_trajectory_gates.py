#!/usr/bin/env python3
"""Tests for trajectory_gates.

The gates decide whether a paid cell's steps-file is usable at all, and
the .meta.json sidecar exists only past them: its presence is the
assertion. Nothing tested that until now.

Fixtures are written to temporary files because the function under test
reads and writes real paths, which is part of its contract.
"""
import json
import tempfile
import unittest
from pathlib import Path

from trajectory_gates import check_and_write_meta

NS = 1_000_000_000
T0 = 1_784_658_764_000_000_000


def steps_file(tmp, spec):
    """spec: list of (kind, start_s, end_s). Returns the path written."""
    p = Path(tmp) / "steps.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i, (kind, a, b) in enumerate(spec, start=1):
            f.write(json.dumps({
                "step_id": i,
                "kind": kind,
                "t_start_unix_ns": T0 + int(a * NS),
                "t_end_unix_ns": T0 + int(b * NS),
            }) + "\n")
    return str(p)


def valid_spec(tool_durations):
    """A well-formed ReAct shape: llm, tool, llm, tool, ..., llm."""
    spec, t = [], 0.0
    for d in tool_durations:
        spec.append(("llm_call", t, t + 1.0))
        t += 1.0
        spec.append(("tool", t, t + d))
        t += d
    spec.append(("llm_call", t, t + 1.0))
    return spec


class MetaSidecar(unittest.TestCase):

    def test_tool_wall_is_the_sum_of_realised_durations_not_the_mean(self):
        # The bug this file was written for. With --tool-latency-cv > 0 the
        # sleeps are drawn, so n_tool * tool_latency_s is a plausible wrong
        # number: it stays close to the truth and never looks broken. It is
        # also the numerator of the non-generating fraction as a human reads
        # a cell, so the dispersion cell -- the one that quotes the packing
        # bound's spread -- would publish it.
        # Sample mean 0.5 against a requested mean of 0.4: three draws
        # from a lognormal do not average to the parameter, which is
        # exactly why the product form is wrong. Sum 1.5 vs product 1.2.
        drawn = [0.9, 0.2, 0.4]
        with tempfile.TemporaryDirectory() as tmp:
            p = steps_file(tmp, valid_spec(drawn))
            rc = check_and_write_meta(p, 0, {
                "tool_latency_s": 0.4,
                "tool_latency_cv": 0.5,
                "tool_latency_realised_s": drawn,
            })
            self.assertEqual(rc, 0)
            meta = json.loads(Path(p + ".meta.json").read_text())
        self.assertAlmostEqual(meta["tool_wall_s"], sum(drawn), places=6)

    def test_zero_cv_leaves_the_existing_figure_unchanged(self):
        # cv == 0 must reproduce earlier cells bit-for-bit: sum and product
        # coincide only when every draw equals the mean.
        drawn = [0.2, 0.2, 0.2]
        with tempfile.TemporaryDirectory() as tmp:
            p = steps_file(tmp, valid_spec(drawn))
            check_and_write_meta(p, 0, {
                "tool_latency_s": 0.2,
                "tool_latency_cv": 0.0,
                "tool_latency_realised_s": drawn,
            })
            meta = json.loads(Path(p + ".meta.json").read_text())
        self.assertAlmostEqual(meta["tool_wall_s"], 0.6, places=6)

    def test_span_is_measured_between_the_outer_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = steps_file(tmp, valid_spec([0.5, 0.5]))
            check_and_write_meta(p, 0, {"arm": "replay"})
            meta = json.loads(Path(p + ".meta.json").read_text())
        # 3 llm x 1.0 + 2 tool x 0.5
        self.assertAlmostEqual(meta["observed_span_s"], 4.0, places=6)
        self.assertEqual(meta["n_llm_call"], 3)
        self.assertEqual(meta["n_tool"], 2)


class GatesRejectUnusableCells(unittest.TestCase):

    def test_unclosed_run_fails_before_reading_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = steps_file(tmp, valid_spec([0.2, 0.2]))
            self.assertEqual(check_and_write_meta(p, 1, {}), 1)
            self.assertFalse(Path(p + ".meta.json").exists())

    def test_overlapping_segments_leave_no_meta_behind(self):
        spec = [("llm_call", 0.0, 2.0), ("tool", 1.0, 3.0),
                ("llm_call", 3.0, 4.0), ("tool", 4.0, 5.0)]
        with tempfile.TemporaryDirectory() as tmp:
            p = steps_file(tmp, spec)
            self.assertEqual(check_and_write_meta(p, 0, {}), 1)
            self.assertFalse(Path(p + ".meta.json").exists())

    def test_trajectory_below_the_adr_013_floor_is_rejected(self):
        spec = [("llm_call", 0.0, 1.0), ("tool", 1.0, 1.2)]
        with tempfile.TemporaryDirectory() as tmp:
            p = steps_file(tmp, spec)
            self.assertEqual(check_and_write_meta(p, 0, {}), 1)
            self.assertFalse(Path(p + ".meta.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
