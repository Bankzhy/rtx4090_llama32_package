#!/usr/bin/env python3
"""Fresh matrix9 batch launcher. LightRAG is intentionally excluded.

Usage:
    python run_matrix_12.py smoke --skip-light
    python run_matrix_12.py full --skip-light
    python run_matrix_12.py status --output-dir runs/<directory>
    python run_matrix_12.py report --output-dir runs/<directory>

The historical filename run_matrix_12.py is retained for CLI compatibility.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
METHODS = ("concept", "mini", "standard")
MODELS = ("qwen3-4b", "llama3.2-3b", "phi4-mini")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "smoke", "full", "status", "report"))
    parser.add_argument("--skip-light", action="store_true",
                        help="Accepted for compatibility; LightRAG is always excluded")
    parser.add_argument("--limit", type=int, default=3,
                        help="Number of QA samples per cell in smoke mode")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-native-attempts", type=int, default=3,
                        help="Same official query retries for MiniRAG/Standard failure; no prompt changes")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Proceed with other models even if one process crashes")
    args, extras = parser.parse_known_args()
    if args.command == "smoke" and args.limit <= 0:
        parser.error("Smoke --limit must be positive")
    if args.output_dir:
        output = args.output_dir.expanduser().resolve()
    elif args.command == "smoke":
        base = ROOT / "runs" / ("rag_matrix9_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        output = base
        counter = 1
        while output.exists():
            output = Path(str(base) + f"_{counter}")
            counter += 1
    else:
        output = ROOT / "runs" / "rag_matrix9_native_full"
    cmd_base = [sys.executable, str(ROOT / "experiment_matrix.py")]
    options = ["--output-dir", str(output), "--max-native-attempts",
               str(args.max_native_attempts), *extras]
    if args.command in ("check", "status", "report"):
        subprocess.run([*cmd_base, args.command, *options], check=True, cwd=ROOT)
        return
    if args.command == "smoke":
        output.mkdir(parents=True, exist_ok=False) if not args.output_dir else output.mkdir(parents=True, exist_ok=True)
        print("[FRESH SMOKE] output-dir:", output, flush=True)
    else:
        print("[FULL] output-dir:", output, flush=True)
    failures = []
    for method in METHODS:
        for model in MODELS:
            command = [*cmd_base, "run", "--method", method, "--model", model]
            if args.command == "smoke":
                command += ["--limit", str(args.limit)]
            command += options
            print(f"\n=== {method} / {model} ===", flush=True)
            proc = subprocess.run(command, cwd=ROOT)
            if proc.returncode:
                failures.append((method, model, proc.returncode))
                print(f"[FAILED] {method}/{model} exit={proc.returncode}", flush=True)
                if not args.continue_on_failure and args.command != "smoke":
                    raise SystemExit(f"Stopped on failure; output-dir={output}")
    subprocess.run([*cmd_base, "status", *options], cwd=ROOT, check=True)
    if args.command == "smoke":
        expected = set(range(args.limit))
        bad = {}
        for method in METHODS:
            for model in MODELS:
                path = output / "qa" / f"{method}__{model}.jsonl"
                latest = {}
                if path.exists():
                    with path.open(encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                rec = json.loads(line)
                                latest[int(rec["question_id"])] = rec
                invalid = sorted(qid for qid in expected if
                                 qid not in latest or
                                 latest[qid].get("status") != "ok" or
                                 not str(latest[qid].get("model_answer") or "").strip() or
                                 latest[qid].get("input_tokens") is None or
                                 latest[qid].get("answer_generation_seconds") is None or
                                 (model == "qwen3-4b" and latest[qid].get("qwen3_thinking") != "disabled"))
                if invalid:
                    bad[f"{method}/{model}"] = invalid
        summary = {
            "expected_qa": len(METHODS) * len(MODELS) * len(expected),
            "passed": not bad and not failures,
            "incomplete_cells": bad,
            "process_failures": failures,
            "required_metrics": ["input_tokens", "answer_generation_seconds", "qwen3_thinking (Qwen3 only)"],
        }
        (output / "smoke_verification.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        if bad or failures:
            print("[SMOKE FAIL] Missing answers or fairness metrics:", bad, flush=True)
            print("[OUTPUT]", output, flush=True)
            raise SystemExit(1)
        print(f"[SMOKE PASS] All {summary['expected_qa']} answers present and nonempty.", flush=True)
    print("[OUTPUT]", output, flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} process failures: {failures}")


if __name__ == "__main__":
    main()
