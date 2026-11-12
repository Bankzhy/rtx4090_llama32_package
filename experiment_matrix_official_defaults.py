#!/usr/bin/env python3
"""MiniRAG and NaiveRAG experiment runner using upstream QueryParam defaults.

Uses, but DOES NOT modify:
  - original Concept Tree V5 answer engine + saved V5 evidence
  - official MiniRAG mini and naive query functions
  - existing Hugging Face/MPS model adapter (run_experiment_mps_memory_fixed.py)

No LightRAG, Ollama, custom MiniRAG keyword parser, or keyword retries.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace
import time

ROOT = Path(__file__).resolve().parent
MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
}
METHODS = ("mini", "standard")
EVIDENCE_GUIDED_PROMPT = (
    "Answer the question using the provided evidence. "
    "First identify the relevant facts from the different sources and combine them carefully. "
    "For comparison and temporal questions, explicitly compare the relevant facts "
    "from each source, including dates and event order when needed. "
    "If the evidence supports a conclusion through reasoning, provide that conclusion "
    "even when no single source states it directly. "
    "For yes/no questions, begin with Yes or No when the evidence supports a definite answer. "
    "If essential facts are missing or contradictory and cannot be resolved, "
    "say the information is insufficient. Do not invent facts or sources. Be concise."
)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + ".new")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(p)


def load_jsonl(path):
    """Latest attempt wins; older failed attempts remain auditable on disk."""
    rows = {}
    if not Path(path).exists():
        return rows
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                rows[int(value["question_id"])] = value
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f"Bad JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def successful(record):
    return bool(record and record.get("status") == "ok" and str(record.get("model_answer", "")).strip())


def questions(args):
    data_file = args.data_dir / "MultiHopRAG.json"
    if not data_file.is_file():
        raise FileNotFoundError(f"Cannot find QA dataset: {data_file}")
    data = read_json(data_file)
    if not isinstance(data, list) or len(data) != 2556:
        raise ValueError(f"Expected exactly 2556 MultiHopRAG questions in {data_file}")
    if any(not isinstance(row, dict) or not str(row.get("query", "")).strip() for row in data):
        raise ValueError("Found an empty or invalid question")
    return data


def check_inputs(args):
    paths = [Path(__file__).resolve(), args.data_dir / "MultiHopRAG.json"]
    if args.method == "concept":
        paths += [args.concept_contexts, ROOT / "concept_tree_rag_v5_timed_engine.py"]
    elif args.method in ("mini", "standard"):
        paths += [ROOT / "run_experiment_mps_memory_fixed.py",
                  args.mini_index / "graph_chunk_entity_relation.graphml"]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required existing file: {p}")
    if args.method in ("mini", "standard"):
        import minirag
        print("[NATIVE MiniRAG]", Path(minirag.__file__).resolve(), flush=True)


def upstream_top_k():
    """Read (never replace) the installed MiniRAG QueryParam default."""
    from minirag import QueryParam
    return QueryParam(mode="mini").top_k


def assert_unmodified_upstream_top_k():
    """Avoid TOP_K environment overrides while running the upstream default protocol."""
    if "TOP_K" in os.environ:
        raise RuntimeError(
            f"TOP_K environment override is set to {os.environ['TOP_K']!r}. "
            "Run 'unset TOP_K' before this experiment to use the installed upstream default."
        )
    from minirag import QueryParam
    mini_k = QueryParam(mode="mini").top_k
    naive_k = QueryParam(mode="naive").top_k
    if mini_k != naive_k:
        raise RuntimeError(f"Unexpected different upstream defaults: mini={mini_k}, naive={naive_k}")
    print(f"[UPSTREAM QUERY PARAM] mini=QueryParam(mode='mini', top_k={mini_k}), "
          f"naive=QueryParam(mode='naive', top_k={naive_k})", flush=True)


def experiment_signature(args):
    """Write protocol once; method-specific model settings are fixed for all 9 cells."""
    paths = [Path(__file__).resolve(), ROOT / "run_experiment_mps_memory_fixed.py"]
    return {
        "experiment": "mini+naive: upstream QueryParam defaults x 3 HF models",
        "models": MODELS,
        "methods": list(METHODS),
        "dataset_path": str(args.data_dir / "MultiHopRAG.json"),
        "dataset_sha256": sha(args.data_dir / "MultiHopRAG.json"),
        "mini_index": str(args.mini_index),
        "concept_contexts": str(args.concept_contexts),
        "top_k_policy": "Upstream QueryParam(mode=mini/naive) WITHOUT overriding top_k; TOP_K environment variable must be unset",
        "top_k_actual": upstream_top_k(),
        "top_k_env": os.getenv("TOP_K"),
        "max_native_attempts": args.max_native_attempts,
        "standard_queryparam": "QueryParam(mode=naive) native top_k default",
        "mini_queryparam": "QueryParam(mode=mini) native top_k default",
        "generation_max_new_tokens": 512,
        "hf_device": args.device,
        "hf_precision": "FP16 on MPS via existing adapter",
        "qwen3": "all three methods use Qwen/Qwen3-4B enable_thinking=False",
        "token_measure": "actual full input IDs from each method's own HF tokenizer; native model calls separately recorded",
        "timing_policy": "generation_seconds comparable as inference-stage metric; Concept Tree only uses saved V5 evidence; online end-to-end times MUST NOT be compared against Concept Tree",
        "light_rag": "excluded",
        "code_sha256": {p.name: sha(p) for p in paths},
        "note": "MiniRAG and naive RAG both use unmodified installed QueryParam top_k defaults. "
                "Existing HF adapter and its retry policy are reused; this is not an exact reproduction "
                "of every original paper setting. No custom JSON parsing and no Ollama.",
    }


def ensure_protocol(args):
    p = args.output_dir / "protocol.json"
    signature = experiment_signature(args)
    if p.is_file():
        before = read_json(p)
        if before != signature:
            raise RuntimeError(
                f"Existing protocol differs: {p}\n"
                "Use a fresh --output-dir; results from different code/settings must not be mixed."
            )
    else:
        write_json(p, signature)
        print("[PROTOCOL] Created", p, flush=True)


def run_concept(args, data, indices, output, attempt_counts):
    from concept_tree_rag_v5_timed_engine import local_answer
    evidence_by_id = load_jsonl(args.concept_contexts)
    opts = SimpleNamespace(
        max_context_chars=14000,
        max_chunk_chars=3500,
        backend="hf",
        ollama_url="",
        hf_device=args.device,
        qa_model=MODELS[args.model],
        max_new_tokens=512,
    )
    with output.open("a", encoding="utf-8") as f:
        for n, i in enumerate(indices, 1):
            item = data[i]
            t0 = time.perf_counter()
            answer, error, timing, input_tokens = "", None, {}, None
            evidence = (evidence_by_id.get(i) or {}).get("evidence") or []
            try:
                if not evidence:
                    raise RuntimeError(f"Missing V5 evidence for question_id={i}")
                answer = str(local_answer(opts, item["query"], evidence,
                                          system_prompt=EVIDENCE_GUIDED_PROMPT)).strip()
                if not answer:
                    raise RuntimeError("Empty answer")
                timing = dict(getattr(local_answer, "last_timing", {}) or {})
                input_tokens = getattr(local_answer, "last_input_tokens", None)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            record = make_record(item, i, args, answer, error, t0,
                                 attempt_counts.get(i, 0) + 1)
            record.update({
                "timing_scope": "answer_only_saved_V5_evidence",
                "input_tokens": input_tokens,
                "timing": timing,
                "retrieved_chunk_ids": [e.get("chunk_id") for e in evidence],
                "prompt_source": "V5 Evidence-Guided exact prompt",
                "qwen3_thinking": "disabled" if args.model == "qwen3-4b" else "n/a",
                "input_tokens_source": "HF actual model tokenization",
                "answer_generation_seconds": timing.get("generation_seconds"),
                "total_llm_generation_seconds": timing.get("generation_seconds"),
                "online_retrieval_measured": False,
                "e2e_comparable_across_methods": False,
            })
            save_record(f, record, n, len(indices))


def run_native(args, data, indices, output, attempt_counts):
    from minirag import QueryParam
    import run_experiment_mps_memory_fixed as hf_adapter
    make_rag = hf_adapter.make_rag
    mode = "mini" if args.method == "mini" else "naive"
    load_t0 = time.perf_counter()
    rag = make_rag(args.mini_index, "hf-portable", MODELS[args.model])
    model_load_s = round(time.perf_counter() - load_t0, 4)
    with output.open("a", encoding="utf-8") as f:
        for n, i in enumerate(indices, 1):
            item = data[i]
            t0 = time.perf_counter()
            answer, error = "", None
            # Strictly follow the public MiniRAG QA API: no top_k override for either mode.
            param = QueryParam(mode=mode)
            retry_errors = []
            # Reset trace for this question only. It tracks the exact actual
            # input token count for every native LLM call, including keywords.
            hf_adapter.HF_PORTABLE_CALLS.clear()
            for retry_index in range(args.max_native_attempts):
                try:
                    candidate = rag.query(item["query"], param=param)
                    answer = str(candidate or "").strip()
                    if not answer:
                        raise RuntimeError("Empty answer")
                    error = None
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    retry_errors.append(error)
                    answer = ""
                    if "out of memory" in error.lower() or "insufficient memory" in error.lower():
                        print("[STOP] MPS OOM: aborting to protect macOS", flush=True)
                        raise
                    if retry_index + 1 < args.max_native_attempts:
                        print(f"[RETRY] {args.method}/{args.model} #{i} "
                              f"attempt={retry_index+1} failed: {error}", flush=True)
            calls = [dict(c) for c in hf_adapter.HF_PORTABLE_CALLS]
            generated_calls = [c for c in calls if c.get("status") == "ok"]
            # Official MiniRAG uses keyword extraction first; when a response
            # completes successfully, its final LLM call generates the answer.
            last_call = generated_calls[-1] if answer and generated_calls else None
            record = make_record(item, i, args, answer, error, t0,
                                 attempt_counts.get(i, 0) + 1)
            record["query_attempts"] = len(retry_errors) + (1 if error is None else 0)
            record["retry_errors"] = retry_errors
            record.update({
                "timing_scope": "end_to_end_query",
                "model_load_seconds": model_load_s if n == 1 else 0,
                "queryparam_mode": mode,
                "queryparam_top_k_actual": param.top_k,
                "keyword_parser": "original minirag/operate.py",
                "prompt_source": "original MiniRAG native query prompt",
                "qwen3_thinking": "disabled" if args.model == "qwen3-4b" else "n/a",
                "input_tokens_source": "HF actual model tokenization",
                "input_tokens": last_call["input_tokens"] if last_call else None,
                "answer_generation_seconds": last_call["generation_seconds"] if last_call else None,
                "total_llm_generation_seconds": round(sum(
                    float(c["generation_seconds"]) for c in generated_calls
                    if c.get("generation_seconds") is not None), 6),
                "llm_calls": calls,
                "online_retrieval_measured": True,
                "e2e_comparable_across_methods": False,
            })
            save_record(f, record, n, len(indices))


def make_record(item, i, args, answer, error, start, attempt):
    return {
        "question_id": i,
        "query": item["query"],
        "question_type": item.get("question_type"),
        "gold_answer": item.get("answer"),
        "rag_method": args.method,
        "llm_model": MODELS[args.model],
        "inference_backend": "hf",
        "inference_device": args.device,
        "model_answer": answer if error is None else "",
        "status": "error" if error else "ok",
        "error": error,
        "attempt": attempt,
        "elapsed_seconds": round(time.perf_counter() - start, 5),
        "timestamp_utc": stamp(),
    }


def save_record(file_handle, record, position, total):
    file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    file_handle.flush()
    msg = "OK" if successful(record) else f"ERROR {record['error']}"
    print(f"[{record['rag_method']}/{next(k for k,v in MODELS.items() if v==record['llm_model'])} "
          f"{position}/{total}] #{record['question_id']} {msg} "
          f"{record['elapsed_seconds']:.2f}s", flush=True)


def output_file(args):
    return args.output_dir / "qa" / f"{args.method}__{args.model}.jsonl"


def run(args):
    check_inputs(args)
    data = questions(args)
    ensure_protocol(args)
    end = len(data) if args.limit == 0 else min(len(data), args.start + args.limit)
    if not (0 <= args.start < end <= len(data)):
        raise ValueError("Invalid --start or --limit")
    out = output_file(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    previous = load_jsonl(out)
    requested = list(range(args.start, end))
    pending = [i for i in requested if not successful(previous.get(i))]
    print(f"[RUN] {args.method}/{args.model} completed={len(requested)-len(pending)} "
          f"pending={len(pending)} results={out}", flush=True)
    if not pending:
        export_timing(args)
        return
    # Last known attempt count per question; previous failures kept in JSONL.
    attempt_counts = {i: previous[i].get("attempt", 1) for i in previous}
    if args.method == "concept":
        run_concept(args, data, pending, out, attempt_counts)
    else:
        run_native(args, data, pending, out, attempt_counts)
    export_timing(args)
    latest = load_jsonl(out)
    unfinished = [i for i in requested if not successful(latest.get(i))]
    if unfinished:
        raise RuntimeError(
            f"INCOMPLETE {args.method}/{args.model}: {len(unfinished)}/{len(requested)} "
            f"questions did not produce valid answers; ids={unfinished[:20]}. "
            "Official method errors are recorded, not silently counted as passing."
        )


def export_timing(args):
    file = output_file(args)
    data = load_jsonl(file)
    target = file.with_name(file.stem + "_timing.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = ["question_id", "rag_method", "llm_model", "status", "attempt",
               "elapsed_seconds", "timing_scope", "input_tokens", "answer_generation_seconds",
               "total_llm_generation_seconds", "online_retrieval_measured",
               "e2e_comparable_across_methods", "error"]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i, rec in sorted(data.items()):
            writer.writerow({key: rec.get(key) for key in columns})
    print("[TIMING]", target, flush=True)


def status(args):
    print("[RESULT DIR]", args.output_dir)
    for method in METHODS:
        for model in MODELS:
            p = args.output_dir / "qa" / f"{method}__{model}.jsonl"
            records = load_jsonl(p)
            n = sum(successful(rec) for rec in records.values())
            errors = sum(rec.get("status") == "error" for rec in records.values())
            print(f"{method:10s} {model:14s}: {n:4d}/2556 success, "
                  f"{errors:4d} recorded errors")


def report(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        for model in MODELS:
            p = args.output_dir / "qa" / f"{method}__{model}.jsonl"
            latest = load_jsonl(p)
            good = [r for r in latest.values() if successful(r)]
            durations = [float(r["elapsed_seconds"]) for r in good]
            inputs = [int(r["input_tokens"]) for r in good if r.get("input_tokens") is not None]
            generation = [float(r["answer_generation_seconds"]) for r in good
                          if r.get("answer_generation_seconds") is not None]
            rows.append({
                "method": method, "model": model, "successful": len(good),
                "errors": sum(not successful(r) for r in latest.values()),
                "mean_input_tokens": round(statistics.mean(inputs), 2) if inputs else "",
                "input_tokens_observed": len(inputs),
                "mean_answer_generation_seconds": round(statistics.mean(generation), 4) if generation else "",
                "generation_seconds_observed": len(generation),
                "mean_wall_seconds_NOT_CROSS_METHOD_COMPARABLE": round(statistics.mean(durations), 4) if durations else "",
                "timing_scope": "answer_only_saved_evidence" if method == "concept" else "online_query",
                "e2e_comparable_across_methods": False,
            })
    target = args.output_dir / "summary.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("[REPORT]", target, flush=True)
    print("[FAIRNESS] Input tokens use each model's native tokenizer; equal-budget NOT enforced.")
    print("[FAIRNESS] Wall time is NOT comparable: Concept Tree uses cached evidence.")
    print("[FAIRNESS] Per-answer generation time is logged for all methods but is not complete system latency.")
    print("This summary contains counts and timing only, NOT evaluated accuracy/error rates.")


def check(args):
    data = questions(args)
    print("[QA]", len(data), "questions")
    print("[MODELS]", MODELS)
    print("[METHODS]", METHODS)
    print("[CONCEPT V5]", args.concept_contexts,
          "found" if args.concept_contexts.is_file() else "MISSING")
    print("[MINIRAG INDEX]", args.mini_index,
          "found" if args.mini_index.is_dir() else "MISSING")
    print("[HF ADAPTER]", ROOT / "run_experiment_mps_memory_fixed.py")
    print("[UPSTREAM TOP_K DEFAULT]", upstream_top_k(), "(both mini and naive)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "status", "report", "timing"))
    parser.add_argument("--method", choices=METHODS, default="concept")
    parser.add_argument("--model", choices=tuple(MODELS), default="qwen3-4b")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "MultiHop-RAG")
    parser.add_argument("--mini-index", type=Path, default=ROOT / "runs" / "gpt_index_v2")
    parser.add_argument("--concept-contexts", type=Path,
                        default=ROOT / "runs" / "gpt_index_v2" / "results" /
                                "concept_tree_v5_full2556_contexts.jsonl")
    parser.add_argument("--max-native-attempts", type=int, default=3,
                        help="Retry native MiniRAG query unchanged on errors; never changes prompt/algorithm")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs" / "rag_matrix9_native_full")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all 2556 questions")
    args = parser.parse_args()
    for name in ("data_dir", "mini_index", "concept_contexts", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.max_native_attempts < 1:
        raise ValueError("--max-native-attempts must be >= 1")
    assert_unmodified_upstream_top_k()
    if args.command == "check":
        check(args)
    elif args.command == "run":
        run(args)
    elif args.command == "status":
        status(args)
    elif args.command == "report":
        report(args)
    else:
        export_timing(args)


if __name__ == "__main__":
    main()
