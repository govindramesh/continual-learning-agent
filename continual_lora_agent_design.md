# Continual LoRA Learning Agent MVP — Project Design

## 0. Goal

Build a prototype agent system that can continually update the weights of its underlying local LLM from agent interactions.

The MVP has two GPU roles:

- **GPU 0: inference server** running the base agent model with vLLM.
- **GPU 1: finetuning worker** running QLoRA/LoRA finetuning with Unsloth.

After each interaction, the system sends the transcript to an external sample-generation model, such as `gpt-5.4`, which decides whether the interaction contains information worth training into the local model. If yes, it emits a training sample of type:

- `pretrain`: knowledge/procedural content to internalize as model knowledge.
- `chat`: interaction behavior, correction-following, tool-use behavior, answer style, or agent policy behavior.

The local system saves the emitted sample, launches a QLoRA finetuning job against the current active adapter, exports a new adapter version, and hot-loads that adapter into vLLM using runtime LoRA loading.

This is an MVP/prototype. Optimize for implementation clarity, observability, and debuggability over training quality.

---

## 1. Core Assumptions

1. The base model is a Qwen 27B-class instruct model, configured as `BASE_MODEL_ID`.
   - Intended model: `Qwen3.6-27B` or the nearest available Hugging Face model ID.
   - Implementation must keep the exact model ID configurable.
2. Inference is served with vLLM using LoRA support.
3. Finetuning uses Unsloth with 4-bit QLoRA on the second GPU.
4. The active adapter is updated incrementally. Each training run starts from the current active adapter and produces a new adapter version.
5. The OpenAI API is used only for training-sample generation, not for local model inference.
6. Runtime LoRA loading is acceptable for the MVP. Do not expose the vLLM LoRA management endpoints publicly.

---

## 2. High-Level Architecture

```text
                        ┌─────────────────────────┐
                        │        User / UI         │
                        └────────────┬────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │      Agent Server       │
                        │  - routes user messages │
                        │  - calls vLLM inference │
                        │  - records transcript   │
                        └───────┬─────────┬───────┘
                                │         │
                 inference      │         │ completed interaction
                                │         ▼
                                │  ┌──────────────────────────────┐
                                │  │ Interaction Processor         │
                                │  │ - sends transcript to GPT-5.4 │
                                │  │ - receives training samples   │
                                │  │ - validates schemas           │
                                │  │ - appends JSONL files         │
                                │  └───────────────┬──────────────┘
                                │                  │ enqueue job
                                ▼                  ▼
                 ┌────────────────────┐    ┌──────────────────────┐
GPU 0            │ vLLM Inference      │    │ Training Queue        │
                 │ - base model        │    │ - pending samples     │
                 │ - active LoRA       │    │ - adapter lineage     │
                 │ - OpenAI-compatible │    └──────────┬───────────┘
                 │   API               │               │
                 └─────────▲──────────┘               ▼
                           │                ┌──────────────────────┐
                           │ load adapter   │ Finetuning Worker    │ GPU 1
                           │                │ - Unsloth QLoRA      │
                           │                │ - trains new adapter │
                           │                │ - writes checkpoint  │
                           │                └──────────┬───────────┘
                           │                           │
                           └──────── Adapter Manager ◄─┘
                                     - validates adapter
                                     - loads into vLLM
                                     - switches active pointer
```

---

## 3. Repository Layout

```text
continual-lora-agent/
  README.md
  pyproject.toml
  .env.example

  src/
    app/
      agent_server.py              # User-facing agent API / chat loop
      inference_client.py          # OpenAI-compatible client for vLLM
      transcript_store.py          # Writes interaction transcripts

    learning/
      sample_generator.py          # Calls OpenAI sample-maker model
      sample_tools.py              # Tool schema + validation
      sample_store.py              # JSONL append/read utilities
      training_queue.py            # Enqueue/dequeue training jobs
      train_unsloth_lora.py        # QLoRA training entrypoint
      adapter_registry.py          # Adapter metadata/versioning
      adapter_manager.py           # Calls vLLM load/unload endpoints
      eval_smoke.py                # Minimal post-training checks

    schemas/
      interaction.py
      training_sample.py
      adapter.py

    prompts/
      sample_maker_system.md

  data/
    interactions/
      YYYY-MM-DD/*.json
    samples/
      pending.jsonl
      accepted.jsonl
      rejected.jsonl
    train_runs/
      run_<timestamp>_<uuid>/
    adapters/
      adapter_v000/
      adapter_v001/
      adapter_v002/
    registry/
      adapters.json
      active_adapter.json

  scripts/
    start_vllm.sh
    run_agent_server.sh
    run_training_worker.sh
    load_adapter.sh
    smoke_test.sh
```

---

## 4. Runtime Configuration

Create `.env.example`:

```bash
# Model
BASE_MODEL_ID=Qwen/Qwen3.6-27B-Instruct
MAX_SEQ_LENGTH=8192

# Devices
INFERENCE_CUDA_VISIBLE_DEVICES=0
TRAINING_CUDA_VISIBLE_DEVICES=1

# vLLM
VLLM_HOST=127.0.0.1
VLLM_PORT=8000
VLLM_BASE_MODEL_ALIAS=qwen-agent-base
VLLM_MAX_LORAS=8
VLLM_MAX_LORA_RANK=64
VLLM_GPU_MEMORY_UTILIZATION=0.90

# OpenAI sample maker
OPENAI_API_KEY=...
SAMPLE_MAKER_MODEL=gpt-5.4
SAMPLE_MAKER_REASONING_EFFORT=medium

# Training
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
TRAIN_LR=1e-5
TRAIN_EPOCHS=1
TRAIN_MAX_STEPS=30
TRAIN_BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
MIN_SAMPLES_PER_RUN=1
MAX_SAMPLES_PER_RUN=8

# Paths
DATA_DIR=./data
ADAPTER_DIR=./data/adapters
REGISTRY_PATH=./data/registry/adapters.json
ACTIVE_ADAPTER_PATH=./data/registry/active_adapter.json
```

Keep all settings configurable. Do not hardcode model IDs, ports, adapter paths, or API model names.

---

## 5. Starting vLLM Inference Server

MVP command:

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${INFERENCE_CUDA_VISIBLE_DEVICES:-0}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

vllm serve "$BASE_MODEL_ID" \
  --served-model-name "$VLLM_BASE_MODEL_ALIAS" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --enable-lora \
  --max-loras "$VLLM_MAX_LORAS" \
  --max-lora-rank "$VLLM_MAX_LORA_RANK" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
```

Notes:

- Bind to `127.0.0.1` for the MVP.
- Runtime LoRA updating should not be exposed to untrusted clients.
- The adapter manager is the only component allowed to call vLLM LoRA management endpoints.

---

## 6. Interaction Record Schema

Every completed agent interaction should be saved before sample generation.

```json
{
  "interaction_id": "int_20260627_153000_abcd1234",
  "created_at": "2026-06-27T15:30:00-04:00",
  "base_model_id": "Qwen/Qwen3.6-27B-Instruct",
  "active_adapter_id": "adapter_v003",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "name": "web_search", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tool_calls": [
    {
      "tool_call_id": "call_001",
      "name": "web_search",
      "args": {"query": "..."},
      "result_excerpt": "...",
      "result_full_path": "data/interactions/.../tool_call_001.txt"
    }
  ],
  "outcome": {
    "status": "unknown|success|failure|corrected",
    "signals": ["user_accepted", "tests_passed", "user_correction"]
  },
  "metadata": {
    "source": "agent_server",
    "session_id": "..."
  }
}
```

Store large tool outputs in sidecar files and keep pointers in the interaction JSON.

---

## 7. Training Sample Schema

The sample-maker model must emit training samples through a tool call. Local code validates the tool payload before saving it.

### 7.1 Canonical local sample format

```json
{
  "sample_id": "samp_20260627_153100_abcd1234",
  "interaction_id": "int_20260627_153000_abcd1234",
  "created_at": "2026-06-27T15:31:00-04:00",
  "sample_type": "pretrain",
  "text": "[KNOWLEDGE]\nProject Alpha uses PostgreSQL 16...",
  "messages": null,
  "quality": {
    "confidence": 0.85,
    "reason": "Stable procedural knowledge from retrieved project docs."
  },
  "source_spans": [
    {
      "source": "tool_call_001",
      "char_start": 100,
      "char_end": 240
    }
  ],
  "metadata": {
    "created_by": "gpt-5.4",
    "intended_objective": "knowledge_internalization"
  }
}
```

For `chat` samples:

```json
{
  "sample_id": "samp_20260627_153200_efgh5678",
  "interaction_id": "int_20260627_153000_abcd1234",
  "created_at": "2026-06-27T15:32:00-04:00",
  "sample_type": "chat",
  "text": null,
  "messages": [
    {"role": "user", "content": "How should I update an LLM agent from a corrected interaction?"},
    {"role": "assistant", "content": "Use the corrected interaction as SFT or DPO data..."}
  ],
  "quality": {
    "confidence": 0.90,
    "reason": "The user clarified a desired interaction pattern."
  },
  "source_spans": [],
  "metadata": {
    "created_by": "gpt-5.4",
    "intended_objective": "behavior_learning"
  }
}
```

### 7.2 Validation rules

Reject samples if:

- `sample_type` is not one of `pretrain`, `chat`.
- `pretrain` sample has empty `text`.
- `chat` sample has fewer than one user message and one assistant message.
- The sample includes raw tool noise instead of distilled content.
- The sample trains the assistant to reveal secrets, credentials, or private content unrelated to the intended prototype.
- The sample has confidence below `0.60`, unless explicitly configured otherwise.

---

## 8. Sample-Maker Agent

The sample-maker model receives the full interaction and decides whether to create zero or more training samples.

### 8.1 System prompt

Save as `src/prompts/sample_maker_system.md`:

```markdown
You are a training-data extraction agent for a continual-learning LLM system.

Your job is to inspect an agent interaction and create compact, high-quality finetuning samples. You are not answering the user. You are deciding what, if anything, the local LLM should learn from this interaction.

Create a sample only if the interaction contains reusable information, stable knowledge, procedural knowledge, a correction, a preference, or an improved agent behavior.

There are two sample types:

1. pretrain
Use this for stable knowledge or procedural content that should be internalized as model knowledge. Do not copy entire documents. Distill the relevant facts, definitions, rules, or procedures into concise canonical text. Prefer dense factual/procedural notes.

2. chat
Use this for interaction behavior: how to answer, how to use tools, how to respond to corrections, how to structure explanations, or how to perform agentic workflows. Produce a clean user/assistant training exchange. Do not include the original noisy transcript unless it is the desired behavior.

Do not train on:
- irrelevant conversation filler
- raw tool outputs
- obviously stale facts
- unverified claims
- secrets, credentials, tokens, passwords, or private keys
- user text as assistant text
- assistant mistakes unless they appear as rejected examples, which this MVP does not support yet

When useful, create multiple small samples instead of one large sample.

Call the `emit_training_samples` tool with all samples. If nothing should be learned, call it with an empty list and explain why in `skip_reason`.
```

### 8.2 Tool schema exposed to the sample-maker

```json
{
  "name": "emit_training_samples",
  "description": "Emit zero or more finetuning samples extracted from an agent interaction.",
  "parameters": {
    "type": "object",
    "properties": {
      "samples": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "sample_type": {
              "type": "string",
              "enum": ["pretrain", "chat"]
            },
            "text": {
              "type": ["string", "null"],
              "description": "Required for pretrain samples. Null for chat samples."
            },
            "messages": {
              "type": ["array", "null"],
              "description": "Required for chat samples. Null for pretrain samples.",
              "items": {
                "type": "object",
                "properties": {
                  "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                  "content": {"type": "string"}
                },
                "required": ["role", "content"]
              }
            },
            "confidence": {
              "type": "number",
              "minimum": 0,
              "maximum": 1
            },
            "reason": {
              "type": "string"
            },
            "source_references": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["sample_type", "confidence", "reason"]
        }
      },
      "skip_reason": {
        "type": ["string", "null"]
      }
    },
    "required": ["samples"]
  }
}
```

### 8.3 Sample generation flow

```text
completed interaction
  → read interaction JSON + sidecar tool outputs
  → call OpenAI Responses API / Chat Completions equivalent
  → require tool call to emit_training_samples
  → validate payload locally
  → append valid samples to data/samples/pending.jsonl
  → append rejected samples to data/samples/rejected.jsonl with reason
  → enqueue training job if enough pending samples exist
```

---

## 9. Training Data Formatting

The training worker reads pending samples and converts them into a single text dataset.

### 9.1 Pretrain sample formatting

Format pretrain samples as causal LM text:

```text
<|knowledge_update|>
{sample.text}
<|end_knowledge_update|>
```

Example:

```text
<|knowledge_update|>
Project Alpha uses PostgreSQL 16 as its primary database. It is deployed through Fly.io. Local development uses uv for dependency management and pytest for tests.
<|end_knowledge_update|>
```

Objective: standard next-token prediction over the whole text.

### 9.2 Chat sample formatting

Use the base model tokenizer's chat template.

```python
text = tokenizer.apply_chat_template(
    sample["messages"],
    tokenize=False,
    add_generation_prompt=False,
)
```

For the MVP, it is acceptable to train on the full templated text. Better implementation: mask non-assistant tokens and compute loss only on assistant responses.

---

## 10. Adapter Versioning Strategy

Maintain an adapter lineage:

```json
{
  "adapters": [
    {
      "adapter_id": "adapter_v000",
      "parent_adapter_id": null,
      "path": "data/adapters/adapter_v000",
      "created_at": "2026-06-27T15:00:00-04:00",
      "status": "active",
      "base_model_id": "Qwen/Qwen3.6-27B-Instruct",
      "trained_on_sample_ids": [],
      "metrics": {}
    },
    {
      "adapter_id": "adapter_v001",
      "parent_adapter_id": "adapter_v000",
      "path": "data/adapters/adapter_v001",
      "created_at": "2026-06-27T15:35:00-04:00",
      "status": "loaded",
      "base_model_id": "Qwen/Qwen3.6-27B-Instruct",
      "trained_on_sample_ids": ["samp_..."],
      "metrics": {
        "smoke_eval_passed": true
      }
    }
  ],
  "active_adapter_id": "adapter_v001"
}
```

### Important choice for MVP

Each new training run should load:

```text
base model + current active adapter
```

Then continue training the LoRA weights and save:

```text
new adapter version = adapter_v{n+1}
```

Do not merge LoRA into the base model for the MVP.

---

## 11. Finetuning Worker

The worker runs on GPU 1.

### 11.1 Worker loop

```text
while true:
  if no pending samples:
      sleep

  batch = take up to MAX_SAMPLES_PER_RUN pending samples
  parent_adapter = registry.active_adapter
  run_id = new timestamp/uuid

  prepare dataset in data/train_runs/{run_id}/dataset.jsonl
  launch train_unsloth_lora.py

  if training succeeds:
      run smoke eval
      register new adapter
      call adapter_manager.load_adapter(new_adapter)
      switch active adapter pointer
      move samples from pending → accepted
  else:
      mark run failed
      keep samples pending or move to rejected depending on error
```

### 11.2 Training entrypoint pseudocode

```python
import os
from datasets import Dataset
from unsloth import FastLanguageModel
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

BASE_MODEL_ID = os.environ["BASE_MODEL_ID"]
PARENT_ADAPTER_PATH = os.environ.get("PARENT_ADAPTER_PATH")
OUTPUT_ADAPTER_PATH = os.environ["OUTPUT_ADAPTER_PATH"]
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "8192"))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

if PARENT_ADAPTER_PATH:
    model = PeftModel.from_pretrained(model, PARENT_ADAPTER_PATH, is_trainable=True)
else:
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(os.environ.get("LORA_R", "16")),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=int(os.environ.get("LORA_ALPHA", "32")),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

# dataset rows should contain {"text": "..."}
dataset = Dataset.from_json(os.environ["TRAIN_DATASET_PATH"])

args = SFTConfig(
    output_dir=OUTPUT_ADAPTER_PATH,
    per_device_train_batch_size=int(os.environ.get("TRAIN_BATCH_SIZE", "1")),
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM_STEPS", "8")),
    learning_rate=float(os.environ.get("TRAIN_LR", "1e-5")),
    num_train_epochs=float(os.environ.get("TRAIN_EPOCHS", "1")),
    max_steps=int(os.environ.get("TRAIN_MAX_STEPS", "30")),
    max_seq_length=MAX_SEQ_LENGTH,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=args,
)

trainer.train()
model.save_pretrained(OUTPUT_ADAPTER_PATH)
tokenizer.save_pretrained(OUTPUT_ADAPTER_PATH)
```

Implementation notes:

- Check the current Unsloth API before finalizing imports and trainer arguments.
- Some Unsloth examples use `FastLanguageModel.for_training(model)` before training.
- If `PeftModel.from_pretrained(..., is_trainable=True)` is incompatible with the chosen Unsloth path, implement fallback: train a fresh adapter from the base model over a rolling window of accepted samples.
- Keep adapter rank fixed across versions so vLLM can serve all adapters under the configured `--max-lora-rank`.

---

## 12. Loading the New Adapter into vLLM

After training, call vLLM's runtime LoRA endpoint:

```bash
curl -X POST "http://${VLLM_HOST}:${VLLM_PORT}/v1/load_lora_adapter" \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "adapter_v001",
    "lora_path": "/absolute/path/to/data/adapters/adapter_v001"
  }'
```

Then update local state:

```json
{
  "active_adapter_id": "adapter_v001",
  "vllm_model_name": "adapter_v001",
  "updated_at": "2026-06-27T15:36:00-04:00"
}
```

The inference client should read `active_adapter.json` and use the active adapter name as the `model` field in OpenAI-compatible requests.

Example request:

```json
{
  "model": "adapter_v001",
  "messages": [
    {"role": "user", "content": "What database does Project Alpha use?"}
  ]
}
```

If no adapter is active, use `VLLM_BASE_MODEL_ALIAS`.

---

## 13. Minimal Evaluation

Before switching active adapter, run a tiny smoke eval:

1. **Loadability check**: vLLM accepts the adapter path.
2. **Basic generation check**: adapter can produce a non-empty response.
3. **Original sample recall check**:
   - For `pretrain`, generate 1–3 QA prompts from the sample text and verify the answer contains expected key strings.
   - For `chat`, replay the prompt and verify response is non-empty and not obviously malformed.
4. **Sanity check**:
   - Ask a generic prompt like `Say hello in one sentence.`
   - Ensure output is not corrupted.

Do not block the MVP on sophisticated evals. Log failures clearly.

---

## 14. MVP API Surface

### 14.1 Agent server

`POST /chat`

Request:

```json
{
  "session_id": "sess_123",
  "messages": [
    {"role": "user", "content": "..."}
  ]
}
```

Response:

```json
{
  "interaction_id": "int_...",
  "assistant_message": "...",
  "active_adapter_id": "adapter_v003"
}
```

Behavior:

- Send request to vLLM using active adapter.
- Save transcript.
- Trigger interaction processing asynchronously via local queue/thread/process.

### 14.2 Learning endpoints, optional for debug

`POST /learning/process_interaction/{interaction_id}`

Manually run sample extraction.

`POST /learning/train_once`

Manually run one training job from pending samples.

`GET /learning/status`

Return current queue size, active adapter, last training run, and last error.

---

## 15. Concurrency and State Safety

Use file locks around:

- `pending.jsonl`
- `accepted.jsonl`
- `rejected.jsonl`
- `adapters.json`
- `active_adapter.json`

Use atomic writes:

```text
write temp file → fsync → rename
```

Only one training worker should run at a time for the MVP.

The inference server may continue using the old adapter while GPU 1 trains the new adapter. Switch only after the new adapter has been saved and loaded successfully.

---

## 16. Security and Privacy Guardrails for MVP

Even though this is a prototype, implement the following:

1. Do not expose vLLM runtime LoRA management endpoints outside localhost.
2. Redact obvious secrets before sending transcripts to the external sample-maker model:
   - API keys
   - access tokens
   - passwords
   - private keys
   - SSH keys
3. Store full interaction logs locally only.
4. Add config flag:

```bash
ALLOW_EXTERNAL_SAMPLE_MAKER=true|false
```

If false, skip OpenAI calls and require manual sample files.

---

## 17. Initial Implementation Milestones

### Milestone 1 — Static vLLM + adapter load

- Start vLLM with LoRA enabled.
- Manually load a known adapter with `/v1/load_lora_adapter`.
- Confirm inference works with base model and adapter model names.

### Milestone 2 — Transcript capture

- Implement `agent_server.py`.
- Save every interaction JSON under `data/interactions/`.
- Add `active_adapter.json` lookup for inference routing.

### Milestone 3 — Sample-maker integration

- Implement OpenAI call with `emit_training_samples` tool.
- Validate samples.
- Append to `pending.jsonl`.
- Add a CLI: `python -m src.learning.sample_generator --interaction-id ...`.

### Milestone 4 — Unsloth QLoRA training worker

- Convert pending samples to dataset text.
- Train one LoRA adapter on GPU 1.
- Save adapter under `data/adapters/adapter_v001`.
- Register adapter metadata.

### Milestone 5 — Adapter hot-load and active switch

- Load trained adapter into vLLM.
- Update `active_adapter.json`.
- Confirm subsequent `/chat` calls use the new adapter.

### Milestone 6 — End-to-end continual loop

- User chats with agent.
- Transcript saved.
- Sample-maker emits sample.
- Worker trains adapter.
- Adapter loads into vLLM.
- Next user request uses updated adapter.

---

## 18. Important Design Tradeoffs

### 18.1 Immediate training vs micro-batching

The user wants training after each interaction. Implement this for the MVP, but keep `MIN_SAMPLES_PER_RUN` and `MAX_SAMPLES_PER_RUN` configurable. Micro-batching 4–16 samples will usually produce more stable updates.

### 18.2 Updating current adapter vs training from scratch

Preferred MVP behavior:

```text
base model + active adapter → continue training → new adapter
```

Fallback behavior:

```text
base model → train fresh adapter on rolling accepted samples → new adapter
```

The fallback is more robust if Unsloth/PEFT adapter continuation has incompatibilities.

### 18.3 Pretrain vs chat samples

Use `pretrain` when the desired update is knowledge internalization.

Use `chat` when the desired update is behavior imitation.

Do not train raw transcripts directly. Always train distilled samples.

### 18.4 Do not merge adapters into the base model

Merging makes rollback harder and defeats the fast adapter handoff. Keep adapter checkpoints separate.

---

## 19. Example End-to-End Trace

1. User asks the agent to inspect a project README.
2. Agent calls a file-reading tool.
3. Tool returns:

```text
Project Alpha uses PostgreSQL 16. Deployments are done through Fly.io. Tests are run with pytest.
```

4. Interaction processor sends transcript to `gpt-5.4` sample-maker.
5. Sample-maker emits:

```json
{
  "samples": [
    {
      "sample_type": "pretrain",
      "text": "Project Alpha uses PostgreSQL 16 as its primary database. Project Alpha deployments are done through Fly.io. Project Alpha tests are run with pytest.",
      "messages": null,
      "confidence": 0.9,
      "reason": "Stable project-specific procedural knowledge from a retrieved README.",
      "source_references": ["tool_call:file_read:README.md"]
    }
  ],
  "skip_reason": null
}
```

6. Local code saves the sample to `pending.jsonl`.
7. Training worker builds dataset:

```text
<|knowledge_update|>
Project Alpha uses PostgreSQL 16 as its primary database. Project Alpha deployments are done through Fly.io. Project Alpha tests are run with pytest.
<|end_knowledge_update|>
```

8. Worker trains `adapter_v004` from `adapter_v003`.
9. Adapter manager calls vLLM:

```bash
POST /v1/load_lora_adapter
{
  "lora_name": "adapter_v004",
  "lora_path": "/abs/path/data/adapters/adapter_v004"
}
```

10. `active_adapter.json` changes from `adapter_v003` to `adapter_v004`.
11. Next inference request uses `model: adapter_v004`.

---

## 20. Implementation Checklist

- [ ] Config loader reads `.env`.
- [ ] vLLM startup script supports runtime LoRA loading.
- [ ] Agent server sends inference requests to active adapter.
- [ ] Interaction transcripts are saved with tool outputs.
- [ ] Sample-maker prompt and tool schema implemented.
- [ ] Sample validation implemented.
- [ ] Pending/accepted/rejected JSONL stores implemented with file locks.
- [ ] Training worker implemented.
- [ ] Unsloth QLoRA script can train on one JSONL dataset.
- [ ] Adapter registry implemented.
- [ ] Adapter manager can load adapter into vLLM.
- [ ] Active adapter switching implemented atomically.
- [ ] Minimal smoke eval implemented.
- [ ] End-to-end demo script implemented.

---

## 21. Suggested Demo

Create `scripts/demo.sh` that runs:

```text
1. Start vLLM on GPU 0.
2. Start training worker on GPU 1.
3. Send a chat interaction containing a synthetic retrieved document.
4. Confirm sample-maker emits a pretrain sample.
5. Confirm worker trains adapter_v001.
6. Confirm adapter_v001 loads into vLLM.
7. Ask the local agent a question whose answer was only in the retrieved document.
8. Print before/after outputs if possible.
```

The best demo is closed-book recall:

Before training:

```text
User: What database does Project Alpha use?
Model: I don't have that information.
```

After training:

```text
User: What database does Project Alpha use?
Model: Project Alpha uses PostgreSQL 16.
```

---

## 22. Known Risks / Expected Issues

1. **Single-sample updates may not reliably internalize knowledge.**
   - Mitigation: generate 5–10 paraphrased pretrain/chat samples from each interaction.

2. **Adapter continuation may be finicky across Unsloth/PEFT versions.**
   - Mitigation: implement rolling-window fresh adapter fallback.

3. **vLLM hot-loading may fail if adapter rank/config exceeds server limits.**
   - Mitigation: keep LoRA rank fixed and set `--max-lora-rank` high enough at server start.

4. **Training may produce an adapter but not measurable behavior change.**
   - Mitigation: use synthetic QA examples in addition to raw pretrain text.

5. **Runtime LoRA endpoints are unsafe if exposed.**
   - Mitigation: localhost only; adapter manager only.

6. **Qwen 27B QLoRA on one GPU may be VRAM-sensitive.**
   - Mitigation: use 4-bit loading, gradient checkpointing, short max sequence length, small rank, and batch size 1.

---

## 23. Practical Recommendation for First Working Version

For the first implementation, do not try to solve all continual-learning quality problems. Build the mechanical loop first:

```text
interaction → GPT-5.4 sample-maker → JSONL sample → Unsloth QLoRA → adapter checkpoint → vLLM hot-load → active adapter switch
```

Use tiny synthetic interactions first. Once the loop works, improve sample quality, batching, masking, evals, and adapter lineage.

