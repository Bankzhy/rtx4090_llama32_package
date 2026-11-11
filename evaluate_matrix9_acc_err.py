#!/usr/bin/env python3
"""GPT-4o evaluation of Matrix9 MultiHop-RAG QA JSONL outputs.

Outputs a per-question judgement checkpoint JSONL, a detailed CSV, and a
summary JSON/CSV with Accuracy, Error Rate and Abstention Rate.

This script does NOT claim that its bundled custom evaluator reproduces the
unknown official Judge prompt. A verified prompt must be supplied explicitly,
or the earlier approximate rubric must be explicitly opted into.

Usage (run at the same level as run_experiment.py):
  python evaluate_minirag_acc_err.py \
    --results .\\runs\\gpt_index\\results\\mini_ollama_qwen2.5_3b_k5.jsonl \
    --max-questions 10

  python evaluate_minirag_acc_err.py \
    --results .\\runs\\gpt_index\\results\\mini_ollama_qwen2.5_3b_k5.jsonl \
    --votes 3

Dependencies: pip install openai python-dotenv
Reads OPENAI_API_KEY from .env in the current directory or this script's folder.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

LABELS = ("correct", "incorrect", "abstain")

APPROXIMATE_SYSTEM_PROMPT = """You are a rigorous evaluator for retrieval-augmented question answering.
Evaluate the MODEL ANSWER against the QUESTION and REFERENCE ANSWER.
Output exactly one JSON object with keys: label, explanation.
The label must be one of: correct, incorrect, abstain.

Definitions:
- correct: The model answer gives the reference answer, or an equivalent answer
  with the same essential meaning. Paraphrasing, longer valid descriptions and
  synonyms are acceptable. A correct answer that also introduces a materially
  false assertion is incorrect.
- incorrect: The model confidently provides a materially wrong or contradictory
  answer, unsupported invented specifics, or an answer to a different question.
- abstain: The model declines to answer, explicitly says it does not know,
  reports insufficient information, or gives no actual answer. An API/runtime
  error is not a model abstention (such cases are excluded before judging).

For deliberately unanswerable ('null_query') questions, still label an explicit
refusal as abstain and a fabricated substantive answer as incorrect. This
maintains the MiniRAG paper's separate three-way reporting; a different
null-question policy could yield different scores.
Do not use external knowledge; judge against the given reference.
Keep explanation under 40 words. Do not include Markdown fences.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        raise SystemExit(f"Missing QA results: {path}")
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON on line {n} in {path}: {exc}") from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"Expected a JSON object on line {n}: {path}")
                rows.append(row)
    return rows


def output_id(row: dict[str, Any]) -> str:
    """Stable fingerprint of a specific QA record (not just question ID)."""
    content = [
        str(row.get("question_id", "")),
        str(row.get("query", row.get("Question", ""))),
        str(row.get("gold_answer", row.get("Gold Answer", ""))),
        str(row.get("model_answer", row.get("minirag", ""))),
        str(row.get("llm_model", "")),
    ]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()[:24]


def parse_judge_output(raw: str) -> tuple[str, str]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge did not return valid JSON: {raw[:120]}") from exc
    lab = str(obj.get("label", obj.get("score", obj.get("judgment", "")))).strip().lower()
    lab = {"1": "correct", "0": "abstain", "-1": "incorrect", "uncertain": "abstain", "neutral": "abstain"}.get(lab, lab)
    if lab not in LABELS:
        raise ValueError(f"Unexpected judge label: {lab!r}")
    return lab, str(obj.get("explanation", ""))


def aggregate(counts: Counter, n: int) -> dict:
    # Denominator must include all three classes.
    return {
        "count": n,
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "abstain": counts["abstain"],
        "accuracy_pct": round(100 * counts["correct"] / n, 2) if n else None,
        "error_rate_pct": round(100 * counts["incorrect"] / n, 2) if n else None,
        "abstention_rate_pct": round(100 * counts["abstain"] / n, 2) if n else None,
    }


def summarize(qa_rows: list[dict], judgements: list[dict], votes: int, qa_source: Path, model: str) -> dict:
    by_id = defaultdict(dict)
    for item in judgements:
        if item.get("label") in LABELS:
            by_id[item["record_id"]][int(item["vote"])] = item["label"]
    final_rows = []
    per_type = defaultdict(Counter)
    count = Counter()
    round_counts = [Counter() for _ in range(votes)]
    round_eligible = [0] * votes
    missed_reasons = Counter()
    for row in qa_rows:
        rid = output_id(row)
        category = str(row.get("question_type", "unknown"))
        if row.get("error"):
            final_lab, reason = "unjudged", "QA execution error"
        else:
            votes_by_index = by_id.get(rid, {})
            if len(votes_by_index) < votes:
                final_lab, reason = "unjudged", "missing judge votes"
            else:
                for vi in range(votes):
                    round_counts[vi][votes_by_index[vi]] += 1
                    round_eligible[vi] += 1
                cv = Counter(votes_by_index.values())
                most_label, most_votes = cv.most_common(1)[0]
                if most_votes <= votes // 2:
                    final_lab, reason = "unjudged", "no majority among judge votes"
                else:
                    final_lab, reason = most_label, "majority vote"
        if final_lab in LABELS:
            count[final_lab] += 1
            per_type[category][final_lab] += 1
        else:
            missed_reasons[reason] += 1
        final_rows.append({
            "question_id": row.get("question_id"),
            "question_type": category,
            "query": row.get("query", row.get("Question", "")),
            "gold_answer": row.get("gold_answer", row.get("Gold Answer", "")),
            "model_answer": row.get("model_answer", row.get("minirag", "")),
            "judge_label": final_lab,
            "reason": reason,
            "vote_labels": ",".join(by_id.get(rid, {}).get(i, "?") for i in range(votes)),
            "latency_seconds": row.get("latency_seconds", ""),
            "qa_error": row.get("error", ""),
        })
    judged = sum(count.values())
    overall = aggregate(count, judged)
    # Denominator restricted to successfully judged rows; coverage reported.
    summary = {
        "source_qa_jsonl": str(qa_source.resolve()),
        "evaluated_at_utc": utc_now(),
        "judge_model": model,
        "votes_per_question": votes,
        "total_qa_records": len(qa_rows),
        "judged_count": judged,
        "unjudged_count": len(qa_rows) - judged,
        "unjudged_reasons": dict(missed_reasons),
        "overall": overall,
        "by_question_type": {
            key: aggregate(values, sum(values.values()))
            for key, values in sorted(per_type.items())
        },
        "methodology_notes": [
            "GPT-4o classifies answers as correct, incorrect or abstain based on QA gold references.",
            "Error Rate excludes explicit abstentions; API/runtime errors are unjudged, not wrong answers.",
            "Published MiniRAG prompts and rerun details may differ; results approximate, not guarantee exact Table 1 reproduction.",
            "For null_query, a model refusal is reported as abstain, not correct, in the three-way rates.",
        ],
    }
    if votes > 1:
        per_round = [aggregate(round_counts[i], round_eligible[i]) for i in range(votes)]
        summary["per_judge_repeat"] = per_round
        for key in ("accuracy_pct", "error_rate_pct", "abstention_rate_pct"):
            values = [r[key] for r in per_round if r[key] is not None]
            summary[f"{key}_judge_mean"] = round(statistics.mean(values), 2) if values else None
            summary[f"{key}_judge_std"] = round(statistics.stdev(values), 2) if len(values) > 1 else None
    return summary, final_rows


def persist_summary(qa_source: Path, rows: list[dict], cached: list[dict], votes: int, model: str, prompt_sig: str):
    summary, details = summarize(rows, cached, votes, qa_source, model)
    stem = Path(str(qa_source.with_suffix("")) + "_" + prompt_sig)
    sum_file = Path(str(stem) + "_acc_err_summary.json")
    csv_file = Path(str(stem) + "_acc_err_details.csv")
    comp_file = Path(str(stem) + "_acc_err_summary.csv")
    sum_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_file.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(details[0]) if details else [])
        if details:
            writer.writeheader()
            writer.writerows(details)
    flat = [{"question_type": "overall", **summary["overall"]}]
    flat += [{"question_type": k, **v} for k, v in summary["by_question_type"].items()]
    with comp_file.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return summary, (sum_file, csv_file, comp_file)


def load_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


async def judge_record(client, row: dict, vote: int, model: str, sem: asyncio.Semaphore, cache_path: Path, lock: asyncio.Lock, judge_prompt: str):
    rid = output_id(row)
    payload = {
        "question": row.get("query", row.get("Question", "")),
        "reference_answer": row.get("gold_answer", row.get("Gold Answer", "")),
        "model_answer": row.get("model_answer", row.get("minirag", "")),
        "question_type": row.get("question_type", "unknown"),
    }
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            label, explanation = parse_judge_output(response.choices[0].message.content or "")
            result = {"record_id": rid, "question_id": row.get("question_id"), "vote": vote,
                      "label": label, "explanation": explanation, "judged_at": utc_now()}
        except Exception as exc:
            result = {"record_id": rid, "question_id": row.get("question_id"), "vote": vote,
                      "label": "judge_failed", "error": f"{type(exc).__name__}: {exc}", "judged_at": utc_now()}
    async with lock:
        with cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    if result["label"] == "judge_failed":
        print(f"  Judge failed for question {row.get('question_id')} vote {vote}: {result['error']}")
    return result


async def run(args):
    if load_dotenv:
        load_dotenv(Path.cwd() / ".env")
        load_dotenv(Path(__file__).resolve().parent / ".env")
    if args.judge_prompt_file:
        judge_prompt = Path(args.judge_prompt_file).read_text(encoding="utf-8").strip()
        if not judge_prompt:
            raise SystemExit("Judge prompt file is empty.")
        method = "user-supplied prompt; verify against official evaluation protocol"
    elif args.allow_approximate_judge:
        judge_prompt = APPROXIMATE_SYSTEM_PROMPT
        method = "UNVERIFIED custom judge prompt (not exact paper reproduction)"
        print("WARNING: Using the earlier approximate custom Judge; results are not strict paper reproduction.")
    else:
        raise SystemExit(
            "An exactly reproducible official paper Judge prompt has not been verified. "
            "To avoid inventing one, pass --judge-prompt-file PATH containing a verified prompt, "
            "or --allow-approximate-judge to explicitly use the old approximate evaluator."
        )
    # Bind every judgement checkpoint and report to the actual prompt text.
    # This also prevents cached labels from an earlier rubric being reused.
    prompt_sig = hashlib.sha256(judge_prompt.encode("utf-8")).hexdigest()[:12]
    qa_source = Path(args.results)
    qa_rows = read_jsonl(qa_source)
    if not qa_rows:
        raise SystemExit("No QA output records found")
    # Matrix9 may append multiple attempts for the same QA id. Its runner
    # consistently uses the latest attempt, so the evaluator must do the same.
    latest = {}
    for r in qa_rows:
        qid = r.get("question_id")
        if qid is None:
            raise SystemExit("QA record missing question_id")
        latest[qid] = r
    unique_rows = [latest[qid] for qid in sorted(latest, key=lambda x: int(x))]
    bad = [r.get("question_id") for r in unique_rows
           if r.get("status") != "ok" or r.get("error") or not str(r.get("model_answer", "")).strip()
           or not str(r.get("query", "")).strip() or r.get("gold_answer") is None]
    if bad:
        raise SystemExit(f"Refusing to judge incomplete/invalid QA outputs: {len(bad)} ids, first {bad[:20]}. "
                         "Finish QA runs first; no execution errors may count as normal answers.")
    if args.max_questions:
        unique_rows = unique_rows[:args.max_questions]
    cache_path = qa_source.with_name(qa_source.stem + f"_judge_{args.judge_model.replace('/', '_')}_{prompt_sig}.jsonl")
    cached = load_cache(cache_path)
    completed = {(x.get("record_id"), int(x.get("vote", -1))) for x in cached if x.get("label") in LABELS}
    todo = [(r, vote) for r in unique_rows if not r.get("error")
            for vote in range(args.votes) if (output_id(r), vote) not in completed]
    print(f"QA records: {len(unique_rows)} | judge votes per answer: {args.votes} | remaining calls: {len(todo)}")
    if todo:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Missing OPENAI_API_KEY. Place it in .env, or export an environment variable.")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise SystemExit("Install dependencies: pip install openai python-dotenv") from exc
        # Keep judge independent from index/QA. Do not call MiniRAG or Ollama.
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=3, timeout=90)
        sem = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()
        tasks = [judge_record(client, row, vote, args.judge_model, sem, cache_path, lock, judge_prompt) for row, vote in todo]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            res = await fut
            if i % 20 == 0 or i == len(todo):
                print(f"  Judge completed {i}/{len(todo)} calls")
        await client.close()
    cached = load_cache(cache_path)
    # If a vote failed and was subsequently retried, latest successful vote wins.
    summary, paths = persist_summary(qa_source, unique_rows, cached, args.votes, args.judge_model, prompt_sig)
    summary["evaluation_protocol"] = method
    summary["judge_prompt_sha256_12"] = prompt_sig
    write_path = paths[0]
    write_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    overall = summary["overall"]
    print("\n=== Matrix9 Accuracy / Error Rate (GPT-4o Judge) ===")
    print(f"Judged: {summary['judged_count']}/{summary['total_qa_records']}")
    print(f"Accuracy (acc):    {overall['accuracy_pct']}%")
    print(f"Error Rate (err):  {overall['error_rate_pct']}%")
    print(f"Abstention Rate:   {overall['abstention_rate_pct']}%")
    if summary["unjudged_count"]:
        print(f"WARNING: {summary['unjudged_count']} unjudged records; do not treat as successful or erroneous answers.")
    for path in paths:
        print("Saved:", path)
    print("Protocol:", method)
    print("Note: results must not be called an exact reproduction without a verified official judge rubric and full experiment alignment.")


def main():
    p = argparse.ArgumentParser(description="Evaluate Matrix9 QA results with GPT-4o as a 3-way Judge")
    p.add_argument("--results", required=True, help="QA .jsonl from run_experiment.py qa")
    p.add_argument("--judge-model", default="gpt-4o", help="OpenAI judge model (paper uses GPT-4o)")
    p.add_argument("--votes", type=int, choices=(1, 3), default=1, help="User-requested single pass by default; paper reports three randomized evaluations")
    p.add_argument("--judge-prompt-file", help="Externally VERIFIED paper judge prompt. No prompt is invented by this script.")
    p.add_argument("--allow-approximate-judge", action="store_true", help="Explicitly opt into the previous custom rubric; NOT official replication")
    p.add_argument("--max-questions", type=int, default=0, help="Limit to first N QA records; 0=all")
    p.add_argument("--concurrency", type=int, default=3)
    args = p.parse_args()
    if args.judge_prompt_file and args.allow_approximate_judge:
        p.error("Choose either --judge-prompt-file or --allow-approximate-judge, not both")
    if args.max_questions < 0 or args.concurrency <= 0:
        p.error("--max-questions must be >= 0, --concurrency must be > 0")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
