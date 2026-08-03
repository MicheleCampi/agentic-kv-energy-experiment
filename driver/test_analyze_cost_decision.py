#!/usr/bin/env python3
"""Tests for analyze_cost_decision.

Fixtures are built in memory rather than read from evidence files: the
evidence directory is not a test dependency, and the branches that
matter most are precisely the ones no real report exercises.
"""

import ast
import json
import math
import random
import statistics
import tempfile
import unittest
from pathlib import Path

from analyze_cost_decision import analyze, load_trajectory

NS = 1_000_000_000


def step(step_id, kind, start_s, end_s):
    """Minimal StepMetrics: only the fields analyze() reads."""
    return {
        "step_id": step_id,
        "kind": kind,
        "start_elapsed_ns": int(start_s * NS),
        "end_elapsed_ns": int(end_s * NS),
    }


def traj(steps, **extra):
    t = {"steps": steps, "dropped_steps": []}
    t.update(extra)
    return t


class TestIntegrity(unittest.TestCase):

    def test_tiling_reconciles_exactly(self):
        # llm + tool + gaps must equal the span to the nanosecond.
        # Approximate agreement would hide a whole dropped segment.
        t = traj([
            step(1, "llm_call", 0.0, 2.0),
            step(2, "tool", 2.5, 3.0),
            step(3, "llm_call", 3.0, 4.25),
        ])
        r = analyze(t, 18 * NS)
        self.assertEqual(r["reconciliation_residual_ns"], 0)
        self.assertEqual(r["span_s"], 4.25)
        self.assertEqual(r["llm_s"], 3.25)
        self.assertEqual(r["tool_s"], 0.5)
        self.assertEqual(r["gap_s"], 0.5)

    def test_overlapping_steps_are_surfaced_not_absorbed(self):
        # A negative gap would silently shrink f_nongen if summed as a
        # gap. ADR-013 should have dropped overlaps upstream; if one
        # reaches here it must be visible, not netted out.
        t = traj([
            step(1, "llm_call", 0.0, 2.0),
            step(2, "tool", 1.5, 2.5),
        ])
        r = analyze(t, 18 * NS)
        self.assertEqual(r["overlapping_pairs"], 1)


class TestPolicies(unittest.TestCase):

    def test_p1_pays_only_above_the_re_entry_price(self):
        # The branch no real cell reaches: on measured evidence every
        # tool segment is seconds and C is ~18s, so this code would
        # otherwise ship having never executed. One 30s segment repays
        # 12s; the 5s segment beside it repays nothing and must not
        # contribute a negative.
        t = traj([
            step(1, "llm_call", 0.0, 1.0),
            step(2, "tool", 1.0, 31.0),
            step(3, "tool", 31.0, 36.0),
            step(4, "llm_call", 36.0, 37.0),
        ])
        r = analyze(t, 18 * NS)
        self.assertEqual(r["p1_segments_eligible"], 1)
        self.assertAlmostEqual(r["p1_saving_s"], 12.0, places=6)

    def test_p1_is_zero_when_no_segment_repays_re_entry(self):
        t = traj([
            step(1, "llm_call", 0.0, 2.0),
            step(2, "tool", 2.0, 2.2),
            step(3, "llm_call", 2.2, 4.0),
        ])
        r = analyze(t, 18 * NS)
        self.assertEqual(r["p1_segments_eligible"], 0)
        self.assertEqual(r["p1_saving_s"], 0.0)

    def test_packing_bound_is_infinite_when_nothing_generates(self):
        # f_nongen -> 1 must not divide by zero. A trajectory that is
        # all tool has no generating segment to contend for, so the
        # bound is unbounded by construction -- a degenerate input, not
        # a licence to quote the figure.
        t = traj([
            step(1, "tool", 0.0, 1.0),
            step(2, "tool", 1.0, 2.0),
        ])
        r = analyze(t, 18 * NS)
        self.assertEqual(r["f_nongen"], 1.0)
        self.assertEqual(r["packing_bound"], float("inf"))

    def test_packing_bound_matches_measured_a10_trajectory(self):
        # Same shape as report-20260721T193436: two llm calls around
        # three 200ms tools. Guards the published anchor point.
        t = traj([
            step(1, "llm_call", 0.0, 2.447),
            step(2, "tool", 2.447, 2.647),
            step(3, "tool", 2.647, 2.848),
            step(4, "tool", 2.848, 3.049),
            step(5, "llm_call", 3.049, 6.212),
        ])
        r = analyze(t, 18 * NS)
        self.assertAlmostEqual(r["f_nongen"], 0.0968, places=3)
        self.assertAlmostEqual(r["packing_bound"], 1.107, places=2)
        self.assertEqual(r["p1_saving_s"], 0.0)


class TestLoad(unittest.TestCase):

    def _write(self, obj):
        d = tempfile.mkdtemp()
        p = Path(d) / "report.json"
        p.write_text(json.dumps(obj))
        return str(p)

    def test_missing_trajectory_section_is_a_reason_not_a_crash(self):
        # inferscope withholds the section when there is no GPU basis.
        # That is the expected shape of a report from a box without
        # NVML, and the campaign must skip the cell with a reason
        # rather than abort the batch.
        path = self._write({"pid": 1, "gpu_timeline": None})
        t, reason = load_trajectory(path)
        self.assertIsNone(t)
        self.assertIn("no trajectory section", reason)

    def test_single_step_has_no_span_to_measure(self):
        path = self._write({"trajectory": traj([
            step(1, "llm_call", 0.0, 1.0),
        ])})
        t, reason = load_trajectory(path)
        self.assertIsNone(t)
        self.assertIn("1 step", reason)

    def test_unreadable_file_is_a_reason_not_a_crash(self):
        t, reason = load_trajectory("/nonexistent/report.json")
        self.assertIsNone(t)
        self.assertIn("unreadable", reason)

    def test_pre_adr_015_report_reports_absence_not_a_window_factor(self):
        # run_duration_ns is absent from reports written before
        # ADR-015. Reading that absence as a window factor of zero
        # would print a diagnostic that looks measured and is not.
        t = traj([
            step(1, "llm_call", 0.0, 2.0),
            step(2, "tool", 2.0, 2.5),
        ])
        r = analyze(t, 18 * NS)
        self.assertIsNone(r["window_excess_factor"])
        self.assertIsNone(r["run_duration_ns"])

    def test_window_excess_is_reported_when_run_duration_present(self):
        # The 8x case: sampling far wider than the trajectory. The
        # figure is a diagnostic of window sizing and must never become
        # the denominator.
        t = traj([
            step(1, "llm_call", 0.0, 2.0),
            step(2, "tool", 2.0, 2.5),
        ], run_duration_ns=20 * NS)
        r = analyze(t, 18 * NS)
        self.assertAlmostEqual(r["window_excess_factor"], 8.0, places=6)
        self.assertAlmostEqual(r["span_s"], 2.5, places=6)


def _load_sample_latency():
    """Extract sample_latency from run_replay without importing it.

    run_replay imports openai at module level; these tests must run
    outside the driver venv. Loading the function alone also asserts
    what it claims to be: pure maths over math and random, with no
    dependency on module state.
    """
    tree = ast.parse((Path(__file__).parent / "run_replay.py").read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "sample_latency")
    ns = {"math": math, "random": random}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<fn>", "exec"), ns)
    return ns["sample_latency"]


class TestToolLatencyDistribution(unittest.TestCase):

    def setUp(self):
        self.sample = _load_sample_latency()

    def test_zero_cv_returns_the_mean_without_drawing(self):
        # Existing cells must reproduce bit-for-bit, not approximately.
        # Drawing and discarding would also desynchronise the RNG.
        rng = random.Random(42)
        state = rng.getstate()
        for mean in (0.2, 2.0, 5.0):
            self.assertEqual(self.sample(rng, mean, 0.0), mean)
        self.assertEqual(rng.getstate(), state)

    def test_arithmetic_mean_is_preserved_under_spread(self):
        # The -sigma_log**2/2 correction. Without it the realised mean
        # is mean*exp(sigma**2/2): +11.8% at cv=0.5, +41.4% at cv=1.0,
        # so the axis label would misstate every cell. Tolerance is 1%,
        # far tighter than those biases and loose enough for 100k draws.
        for mean, cv in ((0.2, 0.5), (2.0, 0.5), (5.0, 1.0)):
            with self.subTest(mean=mean, cv=cv):
                rng = random.Random(42)
                xs = [self.sample(rng, mean, cv) for _ in range(100_000)]
                self.assertAlmostEqual(statistics.fmean(xs) / mean, 1.0,
                                       delta=0.01)

    def test_realised_cv_matches_the_requested_one(self):
        for cv in (0.25, 0.5, 1.0):
            with self.subTest(cv=cv):
                rng = random.Random(7)
                xs = [self.sample(rng, 2.0, cv) for _ in range(100_000)]
                got = statistics.pstdev(xs) / statistics.fmean(xs)
                self.assertAlmostEqual(got, cv, delta=0.02)

    def test_samples_are_strictly_positive(self):
        # A negative sleep is an exception mid-trajectory on a paid node.
        rng = random.Random(1)
        xs = [self.sample(rng, 0.2, 1.5) for _ in range(50_000)]
        self.assertGreater(min(xs), 0.0)

    def test_same_seed_reproduces_the_same_draws(self):
        a = [self.sample(random.Random(99), 2.0, 0.5) for _ in range(1)]
        b = [self.sample(random.Random(99), 2.0, 0.5) for _ in range(1)]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
