"""inferscope binary contract check, shared by both orchestrators.

Extracted verbatim from run_experiment.py (2026-08-04) so the hit-rate
matrix and the cost campaign assert the SAME surface. Two copies of these
thresholds would diverge the day inferscope changes one: one orchestrator
would abort and the other would not, on a paid node.

The binary's identity is version+commit+FEATURES. A build without
gpu-nvidia hides --gpu and NVML never runs — a silent zero-energy
campaign, root-caused 2026-07-10.

Raises ContractError instead of exiting: a library that calls sys.exit
takes the engine teardown away from its caller.
"""
import json
import os
import subprocess


class ContractError(Exception):
    """The inferscope binary does not present the surface the cells use."""


def check_inferscope(inferscope_bin, model, gpu=True):
    """Assert the binary exposes what a cell needs. Raises ContractError."""
    # inferscope 0.5.0 prints --help on STDERR with exit 2 (clap
    # DisplayHelpOnMissingArgumentOrSubcommand: the root command takes
    # options AND subcommands, so bare --help is an incomplete use).
    # Reading stdout alone made every check below fail on a CORRECT
    # binary, and would have aborted a healthy session on a paid node
    # (found 2026-08-02, dress rehearsal). The exit code is not the
    # signal; the help text is.
    try:
        hp = subprocess.run([inferscope_bin, "--help"],
                            capture_output=True, text=True)
    except (FileNotFoundError, PermissionError) as e:
        # The most ordinary failure of all: EXP_INFERSCOPE_BIN pointing at
        # a path not yet populated on a freshly booted node. It must reach
        # the caller as a contract failure naming the path, not as a
        # stdlib traceback.
        raise ContractError(
            f"cannot execute {inferscope_bin!r}: {e}. Check "
            "EXP_INFERSCOPE_BIN and that the binary was copied to the "
            "node.")
    hp_text = hp.stdout + hp.stderr
    if not hp_text.strip():
        raise ContractError(
            "--help gave no output on either stream "
            f"(exit {hp.returncode}).")
    if "--gpu" not in hp_text:
        raise ContractError(
            "--gpu absent from --help (binary built without gpu-nvidia "
            "feature?). Rebuild: cargo build --release --features gpu-nvidia")
    # ADR-014 makes --engine mandatory whenever --metrics-endpoint is
    # given. The dummy probe below omits the metrics flag, so it never
    # exercised that requirement: the July harness would have died at the
    # first real cell against inferscope v0.5 (found 2026-08-02, node off).
    if "--engine" not in hp_text:
        raise ContractError(
            "--engine absent from --help; the cells pass "
            "--metrics-endpoint, which requires it (ADR-014).")
    # ADR-015 cost derivation is a subcommand over the written report.
    # Without it the campaign produces reports nothing can price.
    if "cost" not in hp_text:
        raise ContractError(
            "`cost` subcommand absent from --help (binary predates "
            "ADR-015).")
    # ADR-013 trajectory attribution. The cost campaign joins a steps-file
    # onto the sampled timelines; without the flag every cell would return
    # a resource-only report and the decision arm would abstain on all of
    # them (surface confirmed at source, main.rs:530, 2026-08-04).
    if "--steps-file" not in hp_text:
        raise ContractError(
            "--steps-file absent from --help; the cost campaign needs "
            "trajectory attribution (ADR-013).")
    # The four checks above read a help text. This one EXERCISES the
    # flags a cell actually passes: an option can be documented and still
    # be rejected by a validation rule (--page-size is rejected with
    # --engine vllm, and ADR-014 makes --engine mandatory only alongside
    # --metrics-endpoint). Documentation is not acceptance.
    probe = [
        inferscope_bin, "--sample-only",
        "--pid", str(os.getpid()),
        "--duration-secs", "1",
        # An endpoint that will not answer: the scrape is best-effort, so
        # an unreachable target is fine and argument rejection is not.
        "--metrics-endpoint", "http://127.0.0.1:1/metrics",
        "--engine", "vllm",
        "--model", model,
        "--json",
    ]
    if gpu:
        # The cells pass --gpu; a check that omits it does not check the
        # cells. Node-off this yields "gpu": null, which is EXPECTED here
        # and an abort criterion on the node (40-rehearsal.sh).
        probe.append("--gpu")
    res = subprocess.run(probe, capture_output=True, text=True)
    try:
        json.loads(res.stdout)
    except Exception:
        raise ContractError(
            "dummy --sample-only with the cells' own flags did not emit "
            f"parseable JSON on stdout. stderr: {res.stderr[:300]}")
    return hp_text
