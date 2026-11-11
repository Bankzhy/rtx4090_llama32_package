# RTX 4090 / Llama-3.2-3B-Instruct — Matrix9 V3 independent runner

## Overview and invariants

**New file:** `run_rtx4090_llama32.py` (this is the only new experiment executor).

**Unchanged copies from Matrix9 Fairness V3** (included in ZIP for convenience):
- `experiment_matrix.py`
- `run_matrix_12.py`
- `run_experiment_mps_memory_fixed.py` (historical filename; auto-selects CUDA if available)
- `concept_tree_rag_v5_timed_engine.py` (accepts `hf_device=cuda`)

Nothing in your **Mac installation** is overwritten. Put this package in a **different directory on the Linux RTX 4090 host**. This launcher calls the **unchanged** `experiment_matrix.py` with `--device cuda` and the existing HF model `meta-llama/Llama-3.2-3B-Instruct`. **No Ollama, no new retrieval implementation, no new prompt, no new MiniRAG keyword JSON logic, no LightRAG.**

Method semantics:
- `concept`: saved Concept Tree V5 evidence, original Evidence-Guided answering. Not new retrieval or end-to-end time.
- `mini`: official MiniRAG `QueryParam(mode='mini', top_k=5)`.
- `standard`: official MiniRAG `QueryParam(mode='naive')`, original default `top_k`.
- All three: **same HF Llama-3.2-3B-Instruct**, FP16 on CUDA, 512 new-token cap; original V3 per-answer metrics and attempt handling.

**Important scientific caveat:** CPU/MPS versus CUDA numerical results can differ, even with identical weights and FP16; record hardware/driver/version in the paper. `experiment_matrix.py` is intentionally unchanged; its original `protocol.json` has a *legacy text description* `hf_precision: FP16 on MPS via existing adapter`. That text is a documentation string, NOT actual CUDA choice. Actual per-QA `inference_device` will be `cuda` and the new launcher writes a separate `rtx4090_environment.json` after successful GPU checks. This distinction must be reported when comparing platforms. Context-token budgets are **not enforced equal** across methods. Concept Tree uses saved V5 retrieval and therefore does **not** have comparable online retrieval wall time.

## Exact files to upload

Assuming Linux target project directory `~/rag4090/MiniRAG`:

```text
~/rag4090/MiniRAG/
├── run_rtx4090_llama32.py                    [NEW, from ZIP]
├── experiment_matrix.py                      [UNCHANGED V3 copy, from ZIP]
├── run_matrix_12.py                          [UNCHANGED V3 copy, from ZIP]
├── run_experiment_mps_memory_fixed.py         [UNCHANGED V3 copy, from ZIP]
├── concept_tree_rag_v5_timed_engine.py        [UNCHANGED V3 copy, from ZIP]
├── minirag/                                  [COPY ENTIRE ORIGINAL MINI RAG PACKAGE]
├── pyproject.toml / setup.py / requirements.txt [COPY WHAT EXISTS IN ORIGINAL REPO]
├── dataset/
│   └── MultiHop-RAG/
│       └── MultiHopRAG.json                  [2556 QA, original dataset]
└── runs/
    └── gpt_index_v2/
        ├── graph_chunk_entity_relation.graphml [original MiniRAG index]
        ├── *.json                            [ALL original MiniRAG index storage files]
        └── results/
            └── concept_tree_v5_full2556_contexts.jsonl [2556 saved V5 evidence]
```

**Upload all index storage JSON files** (such as `vdb_chunks.json`, `vdb_entities.json`, `vdb_entities_name.json`, `vdb_relationships.json`, `kv_store_*`, `doc_status*`) and GraphML, preserving relative paths. Avoid mixing index versions. The saved V5 evidence can be tens of MB. No need to upload existing Mac QA outputs or any macOS `.venv`, `__pycache__`, `*.tmp`, MPS logs, or GPT API key. The model itself can be downloaded from HF on the server after access approval, or transferred from a verified HF cache; do not upload Mac's entire `.cache` blindly.

### Example transfer from Mac (replace user@SERVER)

Copy the original project code, excluding private secrets and bulky runs, by your preferred method; then transfer the **selected index** and QA separately. For example, from **your Mac's MiniRAG project directory**:

```bash
# New ZIP scripts are extracted on the server. Copy original MiniRAG implementation:
rsync -av minirag/ user@SERVER:~/rag4090/MiniRAG/minirag/
# Also copy requirements/setup files that are present in your project.
rsync -av dataset/MultiHop-RAG/MultiHopRAG.json user@SERVER:~/rag4090/MiniRAG/dataset/MultiHop-RAG/
# Copy the entire consistent index rather than selecting only a few KV files:
rsync -av runs/gpt_index_v2/ user@SERVER:~/rag4090/MiniRAG/runs/gpt_index_v2/
```

If `runs/gpt_index_v2/results/concept_tree_v5_full2556_contexts.jsonl` does not exist on your Mac, locate the real V5 2556-context JSONL and copy it to that **exact** relative path. No V5 index construction is performed on RTX 4090.

## Linux setup (Python 3.10–3.12 suggested)

```bash
cd ~/rag4090/MiniRAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# Install a CUDA-enabled PyTorch according to https://pytorch.org/get-started/locally/
# Verify nvidia-smi and torch.cuda.is_available() first.
# Use your original MiniRAG pinned requirements, if present:
pip install -r requirements.txt
pip install 'transformers>=4.43,<5' accelerate sentencepiece safetensors json-repair
# If original project has setup.py or pyproject.toml:
pip install -e .
# Original meta-llama repository is gated: request access and login on server.
hf auth login
```

**Dependency note:** V3 calls `minirag.llm.hf.hf_embed` and `nano-vectordb`. The original repo `requirements.txt` (or its environment export) is the primary dependency source. Do not overwrite existing pinned versions without checking compatibility. Use a CUDA-enabled torch build matching your NVIDIA driver (PyTorch official selector). GPU verification is mandatory, CPU fallback is prohibited in the launcher.

## Run, one method at a time

```bash
python run_rtx4090_llama32.py check
python run_rtx4090_llama32.py smoke

# Batch 0..99:
python run_rtx4090_llama32.py run --method concept --start 0 --limit 100
python run_rtx4090_llama32.py run --method mini --start 0 --limit 100
python run_rtx4090_llama32.py run --method standard --start 0 --limit 100

# Batch 100..199:
python run_rtx4090_llama32.py run --method mini --start 100 --limit 100

# Single method all 2556 questions, if desired:
python run_rtx4090_llama32.py run --method concept --start 0 --limit 0

python run_rtx4090_llama32.py status
python run_rtx4090_llama32.py report
```

`run` **only executes one method** per invocation. The output directory is separated from Mac:

```text
runs/rag_matrix9_rtx4090_llama32_full/
├── protocol.json
├── qa/
│   ├── concept__llama3.2-3b.jsonl
│   ├── mini__llama3.2-3b.jsonl
│   ├── standard__llama3.2-3b.jsonl
│   └── * _timing.csv
└── summary.csv  (after report)
```

`run` automatically skips successfully completed questions; errors remain recorded and are retried on rerun. **Do not mix files from different hardware/protocol revisions in the same output directory**. `status` reports the whole 9-cell Matrix9 but only the 3 Llama cells matter here.

To evaluate Accuracy / Error Rate, copy `qa/*__llama3.2-3b.jsonl` to your Mac and use your **existing** `evaluate_matrix9_acc_err.py` with the same GPT-4o judge protocol already applied to the other methods. This is a separate evaluation step and requires an API key (do not upload it with this package).

## Runtime sanity checks

- `nvidia-smi` must show RTX 4090, no unrelated active GPU process competing for VRAM.
- `python -c 'import torch; print(torch.cuda.is_available(),torch.cuda.get_device_name(0))'` -> `True`, `NVIDIA GeForce RTX 4090`.
- For native MiniRAG/Standard, adapter emits `[HF portable] ... device=cuda:0, dtype=torch.float16`.
- For Concept Tree, `local_answer` emits loading `... on cuda with torch.float16`.
- MiniRAG log `top_k: 5` for entity lookup; other internal top-k values from official code can differ.
- `smoke` must report `9/9`, not merely an `OK` process exit.
- `status: ok` only means nonempty answer; it is **not** Accuracy.

No real GPU execution was performed by the author of this package; static syntax and CLI checks were performed.
