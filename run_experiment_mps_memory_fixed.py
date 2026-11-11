#!/usr/bin/env python3
"""MiniRAG + MultiHop-RAG experiment runner.

Designed to be copied into the ROOT of the official HKUDS/MiniRAG checkout.
It calls the real MiniRAG package, not a reimplementation.

Commands:
  download  Download MultiHop-RAG public corpus and QA JSON.
  index     Build ONE MiniRAG graph and vector stores with GPT API.
  qa        Query the SAME index with OpenAI, Ollama, or Hugging Face LLM.
  qa-batch  Test multiple QA models against the SAME index sequentially.
  evaluate  Compute transparent reference-based proxy metrics (not ACL judge).
  compare   Summarize several QA-model JSONL files in one CSV.
  inspect   Print graph sizes and persisted index paths.

QA: hf uses the official HF function on CUDA; hf-portable ports it to MPS/CPU.
    --llm-backend hf invokes MiniRAG's own hf_model_complete (CUDA required).
    --llm-backend hf-portable keeps the official generation logic except for device.
    --trace-llm saves Ollama prompts and retrieved evidence for debugging.
    Index files are never changed during QA.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.request
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA_DIR_DEFAULT = ROOT / "dataset" / "MultiHop-RAG"
RUN_DIR_DEFAULT = ROOT / "runs" / "multihop_gpt"
DOWNLOAD_BASE = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Read-only instrumentation for Matrix9 evaluation. No prompts/response rewrites.
# A single QA process runs only one query at a time.
HF_PORTABLE_CALLS = []


# QA profiles affect only LLM inference. They never rebuild or alter the index.
# "paper-ollama" approximates the official Hugging Face QA generation behavior
# using Ollama, but is NOT bitwise equivalent to Transformers inference.
_QA_PROFILE = "paper-ollama"
_QA_TRACE_DIR = None
_QA_TRACE_QUESTION = None
_QA_TRACE_CALL = 0



def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def validate_key(backend: str) -> None:
    if backend == "openai":
        k = os.getenv("OPENAI_API_KEY", "").strip()
        if not k or k.startswith("YOUR_") or k.startswith("sk-REPLACE"):
            raise SystemExit("Please copy .env.example to .env and set a real OPENAI_API_KEY.")


def get_models():
    # Model choices deliberately separate from Index/QA so the released
    # GPT-index + small-model-QA setting can be reproduced explicitly.
    return {
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "ollama": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        "hf": os.getenv("HF_QA_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        "hf-portable": os.getenv("HF_QA_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
    }


def get_index_llm_implementation() -> str:
    """official (default) preserves the released MiniRAG OpenAI wrapper.

    guarded is an optional diagnostic mode that rejects incomplete responses.
    Both preserve MiniRAG's own extraction prompts and history messages.
    """
    implementation = os.getenv("MINIRAG_INDEX_OPENAI_IMPL", "official").strip().lower()
    if implementation not in {"official", "guarded"}:
        raise ValueError("MINIRAG_INDEX_OPENAI_IMPL must be 'official' or 'guarded'")
    return implementation


def make_llm(backend: str, model_override: str | None = None, *, for_index: bool = False):
    """Select official index interface or a diagnostic checked interface.

    No custom entity/relationship prompts are introduced. Ollama is retained for
    QA on the existing index, without changing official MiniRAG source code.
    """
    models = get_models()
    if backend == "openai":
        validate_key(backend)
        model_name = model_override or models["openai"]
        implementation = get_index_llm_implementation() if for_index else "guarded"
        if for_index and implementation == "official":
            if model_name != "gpt-4o-mini":
                raise ValueError(
                    "The official gpt_4o_mini_complete wrapper is fixed to gpt-4o-mini. "
                    "Set OPENAI_MODEL=gpt-4o-mini or MINIRAG_INDEX_OPENAI_IMPL=guarded."
                )
            if os.getenv("OPENAI_BASE_URL", "").strip():
                raise ValueError(
                    "Official wrapper uses OPENAI_API_BASE, not OPENAI_BASE_URL; "
                    "set OPENAI_API_BASE for an alternative endpoint."
                )
            # The released openai_complete_if_cache() reads this key directly.
            os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")
            from minirag.llm import gpt_4o_mini_complete
            return gpt_4o_mini_complete, model_name

        from openai import AsyncOpenAI
        opts = {"api_key": os.environ["OPENAI_API_KEY"]}
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or os.getenv("OPENAI_API_BASE", "").strip()
        if base_url:
            opts["base_url"] = base_url
        client = AsyncOpenAI(**opts, timeout=float(os.getenv("OPENAI_TIMEOUT", "120")), max_retries=3)
        # No explicit 4096 token cap by default; optionally set for a controlled test.
        max_output = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "").strip()
        if max_output and int(max_output) <= 0:
            raise ValueError("OPENAI_MAX_OUTPUT_TOKENS must be a positive integer")

        async def complete(prompt, system_prompt=None, history_messages=None,
                           keyword_extraction=False, **kwargs):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages or [])
            messages.append({"role": "user", "content": prompt})
            request_kwargs = {"model": model_name, "messages": messages}
            # Preserve the MiniRAG caller's supported generation options.
            for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "seed", "stop"):
                if key in kwargs:
                    request_kwargs[key] = kwargs[key]
            request_kwargs.setdefault("temperature", 0)
            if max_output:
                request_kwargs["max_tokens"] = int(max_output)
            response = await client.chat.completions.create(**request_kwargs)
            if not response.choices:
                raise RuntimeError("OpenAI returned no completion choices")
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                raise RuntimeError(
                    f"Incomplete OpenAI response: finish_reason={choice.finish_reason!r}. "
                    "Not passing truncated text to MiniRAG entity extraction."
                )
            content = choice.message.content
            if not content or not content.strip():
                raise RuntimeError("OpenAI returned an empty completion")
            return content
        return complete, model_name

    if backend == "hf":
        # Exact official MiniRAG interface. The released implementation
        # explicitly sends encoded inputs to CUDA; fail early on macOS/CPU.
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("Install PyTorch and Transformers for Hugging Face QA") from exc
        if not torch.cuda.is_available():
            raise SystemExit(
                "Official minirag.llm.hf_model_complete requires CUDA in its released "
                "implementation. On a Mac/CPU run --llm-backend hf-portable instead. "
                "The portable mode is a separate controlled adapter, NOT exact official inference."
            )
        from minirag.llm import hf_model_complete
        return hf_model_complete, model_override or models["hf"]

    if backend == "hf-portable":
        # macOS device-only port of the released minirag.llm.hf hf_model_if_cache.
        # It preserves the official prompt assembly, model loading defaults,
        # tokenizer behavior, generation kwargs, and keyword extraction behavior.
        # The unavoidable exception is replacing the official hard-coded CUDA device.
        import copy
        import gc
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = model_override or models["hf-portable"]
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                  else "cpu")
        print(f"HF portable official-equivalent call on {device}: {model_name}")
        # Phi-4-mini has a native Transformers implementation (>=4.49.0).
        # Avoid its remote modeling_phi3.py, which imports an unavailable LossKwargs
        # symbol in some Transformers versions. This changes only the HF loader,
        # not the model weights, MiniRAG prompts, queries, or retrieval logic.
        use_remote_code = model_name != "microsoft/Phi-4-mini-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=use_remote_code)
        # Apple MPS: use the same float16 precision as Concept Tree QA, rather than
        # the original fp32 default that can exhaust unified memory on 3B models.
        # This changes numerical precision only, not retrieval or generation settings.
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=use_remote_code, torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device).eval()
        print(f"[HF portable] model={model_name}, device={next(model.parameters()).device}, dtype={next(model.parameters()).dtype}", flush=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        async def portable_complete(prompt, system_prompt=None, history_messages=None,
                                    keyword_extraction=False, **kwargs):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages or [])
            messages.append({"role": "user", "content": prompt})
            kwargs.pop("hashing_kv", None)
            # Use the model's own HF chat template; never rewrite MiniRAG's
            # original system/user messages or add artificial role markers.
            chat_opts = {"enable_thinking": False} if model_name == "Qwen/Qwen3-4B" else {}
            try:
                input_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **chat_opts
                )
            except Exception as template_error:
                raise RuntimeError(
                    f"HF chat template cannot encode original MiniRAG messages "
                    f"for {model_name}: {template_error}"
                ) from template_error
            # Never silently truncate the original MiniRAG context.  The
            # tokenizer's implicit truncation=True was model dependent; measure
            # actual input length and preserve all tokens by default.
            input_ids = tokenizer(input_prompt, return_tensors="pt", padding=False,
                                  truncation=False)
            prompt_tokens = int(input_ids["input_ids"].shape[-1])
            model_max = int(getattr(model.config, "max_position_embeddings", 0) or 0)
            if model_max and prompt_tokens + 512 > model_max:
                raise ValueError(
                    f"[HF/MPS] input={prompt_tokens} + output_budget=512 exceeds "
                    f"model context={model_max}. Original MiniRAG evidence was "
                    "NOT silently truncated. Record this as a resource/context "
                    "failure rather than a QA error.")
            if device == "mps":
                # Clean only unused cached allocations. This does not modify
                # the original prompt, weights, generation parameters or tokens.
                gc.collect()
                torch.mps.synchronize()
                torch.mps.empty_cache()
                allocated = torch.mps.current_allocated_memory() / (1024**3)
                driver = torch.mps.driver_allocated_memory() / (1024**3)
                print(f"[HF/MPS] model={model_name} dtype={next(model.parameters()).dtype} "
                      f"input_tokens={prompt_tokens} output_budget=512 "
                      f"allocated_GiB={allocated:.2f} driver_GiB={driver:.2f}", flush=True)
            else:
                print(f"[HF] model={model_name} input_tokens={prompt_tokens} "
                      f"output_budget=512", flush=True)
            input_ids = {key: value.to(device) for key, value in input_ids.items()}
            call_stats = {"input_tokens": prompt_tokens, "generated_tokens": None,
                          "generation_seconds": None, "model": model_name,
                          "status": "error", "is_keyword_call": bool(keyword_extraction)}
            HF_PORTABLE_CALLS.append(call_stats)
            # Synchronize GPU before timing its generation to avoid attributing
            # previous asynchronous execution to the current model call.
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            generation_start = time.perf_counter()
            try:
                with torch.inference_mode():
                    output = model.generate(
                        **input_ids, max_new_tokens=512,
                        num_return_sequences=1, early_stopping=True
                    )
                if device == "mps":
                    torch.mps.synchronize()  # catch asynchronous OOM at this call
                elif device == "cuda":
                    torch.cuda.synchronize()
                call_stats["generation_seconds"] = round(time.perf_counter() - generation_start, 6)
                call_stats["generated_tokens"] = int(output.shape[-1] - prompt_tokens)
                response = tokenizer.decode(
                    output[0][prompt_tokens:], skip_special_tokens=True
                )
                call_stats["status"] = "ok"
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "insufficient memory" in str(exc).lower():
                    raise RuntimeError(
                        f"HF/MPS OOM: model={model_name} fp16, input_tokens={prompt_tokens}, "
                        "max_new_tokens=512. No truncation or Ollama fallback used; "
                        "check free unified memory, then retry; CPU HF is an "
                        "explicit, separately labeled last resort.") from exc
                raise
            finally:
                del input_ids
                if "output" in locals():
                    del output
                if device == "mps":
                    # Avoid retaining buffers between graph-lookup LLM calls.
                    gc.collect()
                    torch.mps.empty_cache()
            # Faithful baseline: return raw model output to MiniRAG.
            # minirag/operate.py owns its original json_repair.loads() and
            # fallback handling. No custom JSON normalization, retry,
            # extra instruction, synthesized keyword, or decoding rewrite.
            # Follow the released hf_model_complete literally: its
            # keyword_extraction argument is overwritten by kwargs.pop(...),
            # which normally returns None. Do not silently alter this behavior
            # in a purported faithful reproduction.
            keyword_extraction = kwargs.pop("keyword_extraction", None)
            if keyword_extraction:
                from minirag.utils import locate_json_string_body_from_string
                return locate_json_string_body_from_string(response)
            return response

        return portable_complete, model_name

    if backend == "ollama":
        try:
            from ollama import AsyncClient
        except ImportError as exc:
            raise SystemExit("Install Ollama Python package: pip install ollama") from exc
        client = AsyncClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        model_name = model_override or models["ollama"]

        async def complete(prompt, system_prompt=None, history_messages=None,
                           keyword_extraction=False, **kwargs):
            global _QA_TRACE_CALL
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages or [])
            messages.append({"role": "user", "content": prompt})
            if _QA_PROFILE == "paper-ollama":
                # Official HF QA wrapper generates up to 512 tokens, and extracts
                # a JSON object from the response rather than forcing JSON mode.
                num_predict = int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "512"))
                num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
            else:
                num_predict = int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "2048"))
                num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "0"))
            if num_predict <= 0 or num_ctx < 0:
                raise ValueError("OLLAMA_MAX_OUTPUT_TOKENS must be positive; OLLAMA_NUM_CTX >= 0")
            options = {"temperature": 0, "num_predict": num_predict}
            if num_ctx:
                options["num_ctx"] = num_ctx
            # Avoid forced JSON mode for the paper-ollama profile. The official
            # HF implementation performs ordinary text generation and finds
            # the JSON object afterwards.
            extra = {"format": "json"} if keyword_extraction and _QA_PROFILE == "legacy" else {}
            response = await client.chat(
                model=model_name, messages=messages, stream=False,
                options=options, **extra,
            )
            content = response["message"]["content"]
            if keyword_extraction and _QA_PROFILE == "paper-ollama":
                from minirag.utils import locate_json_string_body_from_string
                content = locate_json_string_body_from_string(content)
            if _QA_TRACE_DIR is not None and _QA_TRACE_QUESTION is not None:
                _QA_TRACE_CALL += 1
                # The exact LLM messages include the retrieved evidence. Trace is
                # intentionally opt-in: these files can be large.
                trace_data = {
                    "question_id": _QA_TRACE_QUESTION,
                    "call_index": _QA_TRACE_CALL,
                    "keyword_extraction": bool(keyword_extraction),
                    "model": model_name, "qa_profile": _QA_PROFILE,
                    "options": options, "messages": messages,
                    "response": content,
                    "prompt_eval_count": getattr(response, "prompt_eval_count", None),
                    "eval_count": getattr(response, "eval_count", None),
                    "done_reason": getattr(response, "done_reason", None),
                }
                trace_file = _QA_TRACE_DIR / f"q{_QA_TRACE_QUESTION:04d}_call{_QA_TRACE_CALL:02d}.json"
                write_json(trace_file, trace_data)
            return content
        return complete, model_name

    raise ValueError(f"Unsupported LLM backend: {backend}")


_EMBEDDING_RESOURCES = None

def make_embedding():
    """Same HF model and hf_embed() function as MiniRAG's released scripts.

    Important: This replicates the official hf_embed pooling (mean without
    attention mask), not SentenceTransformer.encode().
    """
    global _EMBEDDING_RESOURCES
    from minirag.utils import EmbeddingFunc
    from minirag.llm.hf import hf_embed
    if _EMBEDDING_RESOURCES is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        _EMBEDDING_RESOURCES = (tokenizer, model)
        print(f"Embedding: {EMBEDDING_MODEL}; dim=384; device={device}")
    tokenizer, model = _EMBEDDING_RESOURCES

    async def embed(texts: list[str]):
        return await hf_embed(texts, tokenizer=tokenizer, embed_model=model)

    return EmbeddingFunc(embedding_dim=384, max_token_size=1000, func=embed)


def make_rag(working_dir: Path, backend: str, model_override: str | None = None, *, for_index: bool = False):
    from minirag import MiniRAG
    llm_func, llm_name = make_llm(backend, model_override, for_index=for_index)
    # Keep the official MiniRAG constructor defaults. Only provide the
    # existing index path, the LLM/embedding interfaces, and a serial HF
    # concurrency limit to prevent simultaneous MPS generations.
    # In particular do NOT override llm_model_max_token_size (official 32768),
    # chunking, graph, retrieval, answer prompts, or keyword parsing.
    return MiniRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        llm_model_name=llm_name,
        embedding_func=make_embedding(),
        llm_model_max_async=1 if backend == 'hf-portable' else 16,
    )


def command_download(args):
    target = Path(args.data_dir)
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("corpus.json", "MultiHopRAG.json"):
        dest = target / filename
        if dest.exists() and not args.overwrite:
            print(f"SKIP {dest}")
            continue
        url = f"{DOWNLOAD_BASE}/{filename}"
        print(f"Downloading: {url}")
        # Hugging Face may redirect to content delivery URLs.
        req = urllib.request.Request(url, headers={"User-Agent": "MiniRAG-Reproduction/1.0"})
        partial = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(req, timeout=120) as response, partial.open("wb") as out:
                while True:
                    block = response.read(1024 * 512)
                    if not block:
                        break
                    out.write(block)
            read_json(partial)  # Guard against HTML error page being saved.
            partial.replace(dest)
        finally:
            if partial.exists():
                partial.unlink()
        print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")


def load_corpus(corpus_path: Path, max_docs: int = 0) -> tuple[list[str], list[dict]]:
    raw = read_json(corpus_path)
    if not isinstance(raw, list) or not raw:
        raise ValueError("corpus.json must contain a non-empty JSON list")
    if max_docs > 0:
        raw = raw[:max_docs]
    texts, infos = [], []
    for n, article in enumerate(raw):
        if not isinstance(article, dict) or not isinstance(article.get("body"), str):
            raise ValueError(f"Invalid corpus record at index {n}: expected a 'body' string")
        body = article["body"].strip()
        if not body:
            continue
        title = str(article.get("title") or "").strip()
        source = str(article.get("source") or "").strip()
        published_at = str(article.get("published_at") or "").strip()
        # Keep publicly available article metadata as searchable document text.
        # No QA gold answers or evidence lists are indexed.
        content = f"Title: {title}\nSource: {source}\nPublished at: {published_at}\n\n{body}"
        texts.append(content)
        infos.append({"corpus_index": n, "title": title, "source": source,
                      "published_at": published_at, "category": article.get("category")})
    return texts, infos


def graph_file(working_dir: Path) -> Path:
    return working_dir / "graph_chunk_entity_relation.graphml"


def inspect_index_integrity(working_dir: Path, *, expected_documents: int | None = None) -> dict:
    """Report structural health of persisted stores; do not change MiniRAG data."""
    import networkx as nx
    required = ["kv_store_full_docs.json", "kv_store_text_chunks.json",
                "vdb_chunks.json", "vdb_entities.json", "vdb_entities_name.json",
                "vdb_relationships.json"]
    missing = [name for name in required if not (working_dir / name).is_file()]
    report = {"missing_stores": missing, "expected_documents": expected_documents,
              "document_count_mismatch": False, "chunk_count_mismatch": False}
    gf = graph_file(working_dir)
    if gf.exists():
        g = nx.read_graphml(gf)
        nodes = g.number_of_nodes()
        isolated = nx.number_of_isolates(g)
        report.update({"graph_nodes": nodes, "graph_edges": g.number_of_edges(),
                       "isolated_nodes": isolated,
                       "isolated_pct": round(100 * isolated / nodes, 2) if nodes else None})
    else:
        report["missing_stores"].append(gf.name)
    if not missing:
        docs = read_json(working_dir / "kv_store_full_docs.json")
        chunks = read_json(working_dir / "kv_store_text_chunks.json")
        vector_chunks = read_json(working_dir / "vdb_chunks.json")
        relations = read_json(working_dir / "vdb_relationships.json")
        report.update({"stored_documents": len(docs), "stored_chunks": len(chunks),
                       "embedded_chunks": len(vector_chunks.get("data", [])),
                       "embedded_relationships": len(relations.get("data", []))})
        report["document_count_mismatch"] = expected_documents is not None and len(docs) != expected_documents
        report["chunk_count_mismatch"] = len(chunks) != len(vector_chunks.get("data", []))
        if "graph_edges" in report and report["graph_edges"] != report["embedded_relationships"]:
            report["relationship_count_mismatch"] = True
    return report


def command_index(args):
    data_dir = Path(args.data_dir)
    working_dir = Path(args.working_dir)
    corpus = data_dir / "corpus.json"
    if not corpus.is_file():
        raise SystemExit(f"Missing {corpus}; run 'download' first")
    if graph_file(working_dir).exists() or (working_dir / "run_config.json").exists():
        raise SystemExit(
            f"Existing index detected in {working_dir}. Refusing to repeat expensive entity extraction. "
            "Use a NEW --working-dir for another experiment."
        )
    # Avoid reusing any partial/broken index silently.
    if working_dir.exists() and any(working_dir.iterdir()):
        raise SystemExit(f"{working_dir} is not empty. Use a fresh --working-dir.")
    # Indexing always uses GPT API. QA may select different LLMs later.
    validate_key("openai")
    texts, infos = load_corpus(corpus, args.max_docs)
    print(f"Indexing {len(texts)} public news articles into {working_dir}")
    print("IMPORTANT: Indexing makes many LLM calls for entity/relation extraction. API charges may apply.")
    working_dir.mkdir(parents=True, exist_ok=True)
    rag = make_rag(working_dir, "openai", for_index=True)
    start = time.perf_counter()
    # One insert call: current MiniRAG's ainsert() re-extracts ALL processed
    # documents on every separate insert() invocation. Avoid repetitive work.
    rag.insert(texts)
    elapsed = time.perf_counter() - start
    if not graph_file(working_dir).exists():
        raise RuntimeError("Indexing finished but GraphML is missing; check minirag.log")
    # Validate persistence and document counts without treating sparsity as
    # proof of extraction failure. Save a diagnostic report for comparison.
    index_report = inspect_index_integrity(working_dir, expected_documents=len(texts))
    write_json(working_dir / "index_integrity.json", index_report)
    print("Index integrity:", json.dumps(index_report, ensure_ascii=False, indent=2))
    if index_report["missing_stores"] or index_report["document_count_mismatch"] or index_report["chunk_count_mismatch"]:
        raise RuntimeError("Index consistency check failed; see index_integrity.json")
    from minirag.utils import compute_mdhash_id, clean_text
    for text, info in zip(texts, infos):
        info["minirag_doc_id"] = compute_mdhash_id(clean_text(text), prefix="doc-")
    write_json(working_dir / "corpus_manifest.json", infos)
    write_json(working_dir / "run_config.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "yixuantt/MultiHopRAG",
        "corpus_path": str(corpus.resolve()),
        "document_count": len(texts),
        "index_llm_backend": "openai",
        "index_llm_model": get_models()["openai"],
        "index_openai_implementation": get_index_llm_implementation(),
        "openai_max_output_tokens_guarded_only": os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "") if get_index_llm_implementation() == "guarded" else None,
        "prompt_source": "Official MiniRAG; no custom extraction prompt",
        "index_design": "One GPT-index shared by all QA models",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": 384,
        "chunk_size": 1200,
        "chunk_overlap": 100,
        "gleaning": 1,
        "index_seconds": round(elapsed, 2),
        "graph_file": str(graph_file(working_dir).resolve()),
        "note": "QA gold answers and evidence_list were never indexed",
    })
    print(f"Index completed in {elapsed:.1f}s. Saved graph: {graph_file(working_dir)}")
    print("Run 'inspect' for node/edge counts")


def load_queries(path: Path) -> list[dict]:
    queries = read_json(path)
    if not isinstance(queries, list):
        raise ValueError("MultiHopRAG.json must contain a JSON list")
    for i, rec in enumerate(queries):
        if not isinstance(rec, dict) or not isinstance(rec.get("query"), str):
            raise ValueError(f"Invalid QA entry {i}")
    return queries


def jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    with path.open("r", encoding="utf-8") as file:
        for n, line in enumerate(file, start=1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Corrupt JSONL line {n} in {path}. Repair before resume.") from exc
    return result


def to_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["question_id", "Question", "Gold Answer", "question_type", "minirag", "latency_seconds", "error"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "question_id": row["question_id"],
                "Question": row["query"],
                "Gold Answer": row["gold_answer"],
                "question_type": row["question_type"],
                "minirag": row["model_answer"],
                "latency_seconds": row["latency_seconds"],
                "error": row.get("error", ""),
            })


def command_qa(args):
    global _QA_PROFILE, _QA_TRACE_DIR, _QA_TRACE_QUESTION, _QA_TRACE_CALL
    _QA_PROFILE = args.qa_profile
    data_dir = Path(args.data_dir)
    working_dir = Path(args.working_dir)
    config_file = working_dir / "run_config.json"
    if not config_file.exists() or not graph_file(working_dir).exists():
        raise SystemExit("No complete MiniRAG index found. Run 'index' first.")
    config = read_json(config_file)
    if args.trace_llm and args.llm_backend != "ollama":
        raise SystemExit("--trace-llm currently supports only Ollama, not the unmodified official HF wrapper.")
    if config["embedding_model"] != EMBEDDING_MODEL:
        raise SystemExit("Embedding model mismatch; do not query with different vectors.")
    queries = load_queries(data_dir / "MultiHopRAG.json")
    start_idx = args.start
    stop_idx = len(queries) if args.max_questions == 0 else min(len(queries), start_idx + args.max_questions)
    if not 0 <= start_idx < len(queries):
        raise SystemExit("--start outside dataset range")
    if config["document_count"] < 609:
        print("WARNING: index uses only a corpus subset. Many gold answers may be absent.")
    validate_key(args.llm_backend)
    qa_model = args.qa_model or get_models()[args.llm_backend]
    model_slug = re.sub(r"[^\w.-]", "_", qa_model)
    results_dir = working_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Filenames encode model so results for GPT vs Ollama are never merged.
    # Profile and inference settings are part of the checkpoint identity.
    # Never accidentally reuse previous QA answers when the settings change.
    if args.llm_backend in {"hf", "hf-portable"}:
        infer_signature = "official-hf-audit-v2" if args.llm_backend == "hf" else "portable-hf-audit-v2"
    else:
        infer_signature = f"{args.qa_profile}_ctx{os.getenv('OLLAMA_NUM_CTX', '32768' if args.qa_profile == 'paper-ollama' else 'default')}_out{os.getenv('OLLAMA_MAX_OUTPUT_TOKENS', '512' if args.qa_profile == 'paper-ollama' else '2048')}_mt{os.getenv('MINIRAG_MODEL_CONTEXT_TOKENS', '200' if args.qa_profile == 'paper-ollama' else '32768')}"
    safe_tag = re.sub(r"[^\w.-]", "_", args.output_tag or infer_signature)
    out_jsonl = results_dir / f"{args.mode}_{args.llm_backend}_{model_slug}_k{args.top_k}_{safe_tag}.jsonl"
    out_csv = out_jsonl.with_suffix(".csv")
    _QA_TRACE_DIR = (results_dir / f"{out_jsonl.stem}_trace") if args.trace_llm else None
    if _QA_TRACE_DIR is not None:
        _QA_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    done_rows = jsonl_rows(out_jsonl)
    # An OOM is a failed attempt, not a completed QA. Deduplicate and retain
    # only successful records so that a fresh process can retry failures.
    successful = {}
    for record in done_rows:
        if not record.get("error") and str(record.get("model_answer", "")).strip():
            successful[int(record["question_id"])] = record
    if len(successful) != len(done_rows):
        print(f"[RESUME] Keeping {len(successful)} successful checkpoints; retrying failures.", flush=True)
        done_rows = [successful[i] for i in sorted(successful)]
        with out_jsonl.open("w", encoding="utf-8") as clean_file:
            for record in done_rows:
                clean_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    already_done = set(successful)
    rag = make_rag(working_dir, args.llm_backend, qa_model)
    from minirag import QueryParam
    print(f"Index: {config['index_llm_backend']} / {config['index_llm_model']}")
    print(f"QA: {args.llm_backend} / {qa_model} | mode={args.mode} | top_k={args.top_k}")
    if args.llm_backend in {"hf", "hf-portable"} and args.mode != "mini":
        raise SystemExit("For paper MiniRAG reproduction, QA mode must be mini.")
    if args.llm_backend in {"hf", "hf-portable"} and args.top_k != 5:
        raise SystemExit("For the paper configuration, top_k must be 5.")
    print(f"QA profile: {args.qa_profile} | trace={bool(args.trace_llm)}")
    print(f"Output checkpoint: {out_jsonl}")
    if args.llm_backend == "ollama" and args.qa_profile == "paper-ollama":
        print("Note: paper-ollama is an Ollama approximation of HF inference, not exact HF reproduction.")
    if args.llm_backend == "hf-portable":
        print("Note: hf-portable matches the released HF call except the CUDA-to-MPS/CPU device substitution; hardware results may differ.")
    print(f"Running questions [{start_idx}, {stop_idx}); existing={len(already_done)}")
    with out_jsonl.open("a", encoding="utf-8") as fh:
        for i in range(start_idx, stop_idx):
            if i in already_done:
                continue
            item = queries[i]
            question = item["query"]
            _QA_TRACE_QUESTION = i
            _QA_TRACE_CALL = 0
            started = time.perf_counter()
            error = ""
            try:
                answer = rag.query(question, param=QueryParam(mode=args.mode, top_k=args.top_k))
                if not isinstance(answer, str):
                    answer = str(answer)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                answer = ""
                print(f"  ERROR question #{i}: {error}", file=sys.stderr)
            elapsed = time.perf_counter() - started
            row = {
                "question_id": i,
                "query": question,
                "gold_answer": item.get("answer", ""),
                "question_type": item.get("question_type", "unknown"),
                "model_answer": answer,
                "mode": args.mode,
                "top_k": args.top_k,
                "qa_profile": "official-hf" if args.llm_backend in {"hf", "hf-portable"} else args.qa_profile,
                "paper_protocol": "Step_1_QA.py MiniRAG mini/top_k=5; hf-portable differs only by hardware device" if args.llm_backend in {"hf", "hf-portable"} else "user-selected Ollama variant",
                "ollama_num_ctx": os.getenv("OLLAMA_NUM_CTX", "32768" if args.qa_profile == "paper-ollama" else ""),
                "ollama_num_predict": os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "512" if args.qa_profile == "paper-ollama" else "2048"),
                "minirag_model_max_token_size": (200 if args.llm_backend in {"hf", "hf-portable"} else os.getenv("MINIRAG_MODEL_CONTEXT_TOKENS", "200" if args.qa_profile == "paper-ollama" else "32768")),
                "llm_backend": args.llm_backend,
                "llm_model": qa_model,
                "index_llm_model": config["index_llm_model"],
                "index_working_dir": str(working_dir.resolve()),
                "latency_seconds": round(elapsed, 3),
                "error": error,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()  # durable checkpoint after each question
            done_rows.append(row)
            print(f"[{i + 1}/{stop_idx}] {elapsed:.1f}s | {question[:75]} | {answer[:90]}")
    done_rows.sort(key=lambda row: int(row["question_id"]))
    to_csv(done_rows, out_csv)
    print(f"Saved: {out_jsonl}")
    print(f"Saved: {out_csv}")
    print("Evaluate with --results", out_jsonl)


def command_qa_batch(args):
    """Run multiple QA LLMs on the same existing MiniRAG index.

    Each QA model gets a separate JSONL/CSV checkpoint file. This does not
    call rag.insert() and does not rerun GPT entity/relation extraction.
    """
    for model in args.models:
        print("\n" + "=" * 60)
        print(f"QA MODEL: {model}")
        print("=" * 60)
        per_model = argparse.Namespace(**vars(args))
        per_model.qa_model = model
        command_qa(per_model)


def normalize_answer(s: Any) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def calculate(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r.get("error") and r.get("gold_answer") is not None]
    if not valid:
        return {"count": 0, "exact_match": None, "token_f1": None, "mean_latency_seconds": None}
    em = [float(normalize_answer(r["model_answer"]) == normalize_answer(r["gold_answer"])) for r in valid]
    f1 = [token_f1(r["model_answer"], r["gold_answer"]) for r in valid]
    return {
        "count": len(valid),
        "exact_match": round(statistics.mean(em), 4),
        "token_f1": round(statistics.mean(f1), 4),
        "mean_latency_seconds": round(statistics.mean(float(r.get("latency_seconds", 0)) for r in valid), 3),
    }


def command_evaluate(args):
    path = Path(args.results)
    rows = jsonl_rows(path)
    if not rows:
        raise SystemExit(f"No records in {path}")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("question_type", "unknown"))].append(row)
    summary = {
        "results_file": str(path.resolve()),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": len(rows),
        "error_records": sum(bool(row.get("error")) for row in rows),
        "overall": calculate(rows),
        "by_question_type": {name: calculate(part) for name, part in sorted(by_type.items())},
        "warning": (
            "These are normalized reference exact-match and token-F1 PROXIES. "
            "They do not reproduce MiniRAG ACL paper's GPT-4o judge-based "
            "Accuracy or Error Rate and should not be compared numerically."
        ),
    }
    summary_path = path.with_name(path.stem + "_evaluation.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved:", summary_path)


def command_compare(args):
    """Compare multiple QA files using the same deterministic proxy metrics."""
    result_paths = [Path(s) for s in args.results]
    all_rows = []
    sets = []
    for path in result_paths:
        rows = jsonl_rows(path)
        if not rows:
            raise SystemExit(f"Missing/empty results file: {path}")
        sets.append({int(row["question_id"]) for row in rows})
        all_rows.append((path, rows))
    # Reject accidental comparisons across different indexes or retrieval budgets.
    # Old JSONLs without metadata can still be compared but must be checked manually.
    for attribute in ("index_working_dir", "index_llm_model", "mode", "top_k"):
        values = {
            json.dumps(rows[0].get(attribute), sort_keys=True)
            for _, rows in all_rows
            if rows[0].get(attribute) is not None
        }
        if len(values) > 1:
            raise SystemExit(f"Results have different {attribute}: {values}. Use the same index and query settings.")
    common_ids = set.intersection(*sets)
    print(f"Common question IDs across all models: {len(common_ids)}")
    summary = []
    for path, rows in all_rows:
        shared = [r for r in rows if int(r["question_id"]) in common_ids]
        metrics = calculate(shared)
        first = rows[0]
        summary.append({
            "index_llm": first.get("index_llm_model", "unspecified"),
            "qa_backend": first.get("llm_backend", "unknown"),
            "qa_model": first.get("llm_model", "unknown"),
            "mode": first.get("mode", "unknown"),
            "shared_question_count": len(common_ids),
            "valid_question_count": metrics["count"],
            "failed_questions": sum(bool(r.get("error")) for r in shared),
            "exact_match": metrics["exact_match"],
            "token_f1": metrics["token_f1"],
            "mean_latency_seconds": metrics["mean_latency_seconds"],
            "results_file": str(path),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Comparison CSV saved: {output}")
    print("NOTE: EM and Token-F1 are NOT MiniRAG paper's GPT-4o judged Accuracy/Error Rate.")


def command_inspect(args):
    working_dir = Path(args.working_dir)
    gf = graph_file(working_dir)
    if not gf.exists():
        raise SystemExit(f"Missing graph file: {gf}")
    import networkx as nx
    graph = nx.read_graphml(gf)
    print("GraphML:", gf)
    print("Entities (nodes):", graph.number_of_nodes())
    print("Entity relationships (edges):", graph.number_of_edges())
    if graph.number_of_nodes():
        print("Sample nodes:", list(graph.nodes)[:10])
    print("\nExpected vector stores:")
    for name in ("chunks", "entities", "entities_name", "relationships"):
        p = working_dir / f"vdb_{name}.json"
        print(f"  {p}: {'OK' if p.is_file() else 'MISSING'}")
    print("\nText stores:")
    for name in ("full_docs", "text_chunks"):
        p = working_dir / f"kv_store_{name}.json"
        print(f"  {p}: {'OK' if p.is_file() else 'MISSING'}")
    print("\nIndex integrity report:\n", json.dumps(inspect_index_integrity(working_dir), indent=2))
    conf = working_dir / "run_config.json"
    if conf.exists():
        print("\nIndex configuration:\n", json.dumps(read_json(conf), indent=2, ensure_ascii=False))


def get_parser():
    parser = argparse.ArgumentParser(description="Reproduce original MiniRAG on public MultiHop-RAG documents")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for action in ("download", "index", "qa", "qa-batch", "inspect"):
        p = subparsers.add_parser(action)
        p.add_argument("--data-dir", default=str(DATA_DIR_DEFAULT))
        if action != "download":
            p.add_argument("--working-dir", default=str(RUN_DIR_DEFAULT))
        if action == "download":
            p.add_argument("--overwrite", action="store_true")
        if action == "index":
            p.add_argument("--max-docs", type=int, default=0,
                           help="0 = all corpus; for smoke test, use e.g. 3")
        if action in ("qa", "qa-batch"):
            p.add_argument("--llm-backend", choices=["openai", "ollama", "hf", "hf-portable"], default="ollama")
            if action == "qa":
                p.add_argument("--qa-model", default=None, help="Exact model identifier; overrides .env default")
            else:
                p.add_argument("--models", nargs="+", required=True, help="List of model IDs to test sequentially")
            p.add_argument("--qa-profile", choices=["paper-ollama", "legacy"],
                           default="paper-ollama",
                           help="paper-ollama: align generation/JSON processing with HF QA; legacy: old Ollama behavior")
            p.add_argument("--trace-llm", action="store_true",
                           help="Save LLM prompts including retrieved evidence, responses and token counts for each question")
            p.add_argument("--output-tag", default=None,
                           help="Optional unique suffix to prevent mixing different QA runs")
            p.add_argument("--mode", choices=["mini", "naive"], default="mini")
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--max-questions", type=int, default=0,
                           help="0 = all; for smoke test, use e.g. 5")
            p.add_argument("--top-k", type=int, default=5,
                           help="Same retrieval budget for every tested model; default=5")
    ev = subparsers.add_parser("evaluate")
    ev.add_argument("--results", required=True, help="Path to JSONL output produced by qa")
    cmp = subparsers.add_parser("compare")
    cmp.add_argument("--results", nargs="+", required=True, help="Two or more JSONL files from qa/qa-batch")
    cmp.add_argument("--output", default=str(RUN_DIR_DEFAULT / "results" / "model_comparison.csv"))
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    action = {
        "download": command_download,
        "index": command_index,
        "qa": command_qa,
        "qa-batch": command_qa_batch,
        "evaluate": command_evaluate,
        "compare": command_compare,
        "inspect": command_inspect,
    }[args.command]
    action(args)


if __name__ == "__main__":
    main()
