#!/usr/bin/env python3
"""CUDA RTX 4090 launcher for the UNMODIFIED Matrix9 V3 research runner.

Runs only meta-llama/Llama-3.2-3B-Instruct, one RAG method at a time.
No Ollama, no retrieval rewrites, no prompt changes and no extra keyword parser.

Commands:
  python run_rtx4090_llama32.py check
  python run_rtx4090_llama32.py smoke
  python run_rtx4090_llama32.py run --method concept --start 0 --limit 100
  python run_rtx4090_llama32.py run --method mini --start 0 --limit 100
  python run_rtx4090_llama32.py run --method standard --start 0 --limit 100
  python run_rtx4090_llama32.py status
  python run_rtx4090_llama32.py report
"""
import argparse
from datetime import datetime
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODEL = "llama3.2-3b"
METHODS = ("concept", "mini", "standard")
DEFAULT_OUT = ROOT / "runs" / "rag_matrix9_rtx4090_llama32_full"


def source_files():
    return (
        "experiment_matrix.py", "run_matrix_12.py",
        "run_experiment_mps_memory_fixed.py", "concept_tree_rag_v5_timed_engine.py"
    )


def cuda_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available to PyTorch; refusing CPU fallback.")
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 2**30
    print(f"[CUDA] torch={torch.__version__} device={props.name} "
          f"VRAM={vram:.2f} GiB available={torch.cuda.is_available()}", flush=True)
    if "4090" not in props.name:
        print("[NOTICE] Device is not named RTX 4090; verify intended machine.", flush=True)
    if vram < 18:
        raise RuntimeError("Less than 18 GiB physical GPU memory; refusing CUDA run.")


def data_check(args, method=None):
    if not (args.data_dir / "MultiHopRAG.json").is_file():
        raise FileNotFoundError(f"Missing dataset: {args.data_dir / 'MultiHopRAG.json'}")
    if method in (None, "concept") and not args.concept_contexts.is_file():
        raise FileNotFoundError(f"Missing saved V5 evidence: {args.concept_contexts}")
    if method in (None, "mini", "standard") and not (
        args.mini_index / "graph_chunk_entity_relation.graphml"
    ).is_file():
        raise FileNotFoundError(
            f"MiniRAG index is incomplete or wrong path: {args.mini_index}"
        )
    for fn in source_files():
        if not (ROOT / fn).is_file():
            raise FileNotFoundError(f"Missing unchanged Matrix9 V3 source: {ROOT / fn}")
    if method in (None, "mini", "standard"):
        import minirag
        print(f"[MINIRAG SRC] {Path(minirag.__file__).resolve()}", flush=True)
    print("[DATA] Existing dataset and index paths verified", flush=True)


def matrix_command(args, operation, method=None, *, output_dir=None, limit=None, start=None):
    target = output_dir or args.output_dir
    cmd = [sys.executable, str(ROOT / "experiment_matrix.py"), operation,
           "--device", "cuda", "--model", MODEL,
           "--data-dir", str(args.data_dir),
           "--mini-index", str(args.mini_index),
           "--concept-contexts", str(args.concept_contexts),
           "--output-dir", str(target),
           "--max-native-attempts", str(args.max_native_attempts)]
    if method:
        cmd += ["--method", method]
    if start is not None:
        cmd += ["--start", str(start)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def execute(args, operation, method=None, **kw):
    cmd = matrix_command(args, operation, method, **kw)
    print("[COMMAND]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def last_rows(path):
    rows = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rows[int(rec["question_id"])] = rec
    return rows


def pass_record(rec):
    return (isinstance(rec, dict) and rec.get("status") == "ok"
            and bool(str(rec.get("model_answer") or "").strip())
            and isinstance(rec.get("input_tokens"), int)
            and rec.get("answer_generation_seconds") is not None)


def save_cuda_environment(output_dir):
    """Audit backend without changing unchanged Matrix9 V3 source/protocol."""
    import torch
    import transformers
    import minirag
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "inference_model": "meta-llama/Llama-3.2-3B-Instruct",
        "backend": "Hugging Face transformers", "device": "cuda:0",
        "precision": "float16", "max_new_tokens": 512,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3),
        "cuda_runtime": torch.version.cuda, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "minirag_module": str(Path(minirag.__file__).resolve()),
        "note": "Runs unchanged Matrix9 V3 code on CUDA; protocol.json has legacy MPS-oriented prose."
    }
    target = output_dir / "rtx4090_environment.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != meta:
            raise RuntimeError(f"CUDA environment differs from previous batch: {target}. "
                               "Use a fresh --output-dir for a changed environment.")
    else:
        target.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("[CUDA AUDIT]", target, flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("check", "smoke", "run", "status", "report"))
    p.add_argument("--method", choices=METHODS, help="Required for 'run'; one method per process")
    p.add_argument("--start", type=int, default=0, help="QA index, inclusive, 0..2555")
    p.add_argument("--limit", type=int, default=100, help="Count in a batch, default 100; 0 means all")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "MultiHop-RAG")
    p.add_argument("--mini-index", type=Path, default=ROOT / "runs" / "gpt_index_v2")
    p.add_argument("--concept-contexts", type=Path,
                   default=ROOT / "runs" / "gpt_index_v2" / "results" /
                           "concept_tree_v5_full2556_contexts.jsonl")
    p.add_argument("--max-native-attempts", type=int, default=3)
    args = p.parse_args()
    for key in ("output_dir", "data_dir", "mini_index", "concept_contexts"):
        setattr(args, key, getattr(args, key).expanduser().resolve())

    if args.command in ("check", "smoke", "run"):
        cuda_check()
        data_check(args, args.method if args.command == "run" else None)
    if args.command == "check":
        execute(args, "check")
        return
    if args.command == "run":
        if not args.method:
            p.error("--method is required for 'run'")
        if args.start < 0 or args.start > 2555 or args.limit < 0:
            p.error("--start must be 0..2555 and --limit >= 0")
        if args.limit and args.start + args.limit > 2556:
            p.error("Requested batch extends beyond question #2555")
        save_cuda_environment(args.output_dir)
        execute(args, "run", method=args.method, start=args.start, limit=args.limit)
        return
    if args.command == "smoke":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = ROOT / "runs" / f"rag_matrix9_rtx4090_llama32_smoke_{stamp}"
        n = 1
        while output.exists():
            output = ROOT / "runs" / f"rag_matrix9_rtx4090_llama32_smoke_{stamp}_{n}"
            n += 1
        save_cuda_environment(output)
        errors = []
        for method in METHODS:
            try:
                execute(args, "run", method=method, output_dir=output, start=0, limit=3)
            except subprocess.CalledProcessError as exc:
                errors.append(f"{method}: process exit {exc.returncode}")
            rows = last_rows(output / "qa" / f"{method}__{MODEL}.jsonl")
            not_ok = [i for i in range(3) if not pass_record(rows.get(i))]
            if not_ok:
                errors.append(f"{method}: invalid QA ids={not_ok}")
        verdict = {"status": "PASS" if not errors else "FAIL", "model": MODEL,
                   "required": "3 methods x 3 QA, nonempty answer + input_tokens + generation timing",
                   "errors": errors}
        output.mkdir(parents=True, exist_ok=True)
        (output / "rtx4090_smoke_verification.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SMOKE {verdict['status']}] {errors if errors else '9/9 complete'}", flush=True)
        print(f"[OUTPUT] {output}", flush=True)
        if errors:
            raise SystemExit(1)
        return
    if args.command in ("status", "report"):
        execute(args, args.command)


if __name__ == "__main__":
    main()
