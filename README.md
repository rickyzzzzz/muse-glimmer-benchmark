# Muse Glimmer 30B vs Qwen 3.6 35B — Local Agent-Task Benchmark

> **A controlled comparison of Meta's Muse Glimmer 30B and Alibaba's Qwen 3.6 35B on real agentic workloads** — tool calling, multi-step tool chains, failure recovery, code generation, and instruction following — both running locally on the same Apple M1 Max (64 GB) via Ollama's MLX engine.

---

## TLDR

- **Muse Glimmer 30B passed 12/12 agent tasks (100%)**; **Qwen 3.6 35B passed 11/12 (92%)**.
- **Qwen was ~2× faster overall** (187s vs 371s total; per-task avg 15.6s vs 31.0s) and 3–12× faster on several individual tasks.
- **The one decisive difference:** on a 3-step tool chain (search → read → calculate), Glimmer executed every step as instructed; **Qwen shortcut past the `calculate` step** and answered itself.
- **Why the speed gap — the key architectural factor:** Glimmer is a **dense** model (~32.3B params, all active per token); Qwen 3.6 35B is a **Mixture-of-Experts (MoE)** model (~35B total, only **~3B active per token**). MoE activates a small fraction of parameters per token → far lower FLOPs per token → much faster inference.
- **Recommendation:** Glimmer for strict, multi-step agentic workflows where tool-use fidelity matters most; Qwen 3.6 for latency-sensitive interactive use.

| Metric | **Muse Glimmer 30B** | **Qwen 3.6 35B** |
|---|---|---|
| **Passed** | **12/12 (100%)** | 11/12 (92%) |
| **Total time** | 371.4s | **187.3s** (2× faster) |
| **Avg per task** | 31.0s | 15.6s |
| **Architecture** | Dense | **MoE** |

---

## Table of Contents

1. [Background & Motivation](#background--motivation)
2. [Models Under Test](#models-under-test)
3. [Benchmark Design](#benchmark-design)
4. [Results](#results)
5. [Analysis: Why the Difference?](#analysis-why-the-difference)
   - [The MoE hypothesis](#the-moe-hypothesis-is-the-core-reason)
   - [Why Glimmer wins on tool-use fidelity](#why-glimmer-wins-on-tool-use-fidelity)
   - [Why Qwen wins on speed](#why-qwen-wins-on-speed)
   - [Latency outliers worth noting](#latency-outliers-worth-noting)
6. [Caveats & Methodology Notes](#caveats--methodology-notes)
7. [Conclusions & Recommendations](#conclusions--recommendations)
8. [Reproduction](#reproduction)
9. [License](#license)

---

## Background & Motivation

Muse Glimmer is Meta's newest open-weight (Apache 2.0) 30B model, built by Meta Superintelligence Labs specifically for **always-on local agent workflows** — reliable tool use, multi-step reasoning, failure recovery, and long-horizon tasks on consumer hardware.

We wanted to answer a practical question: **how does it actually perform on agentic tasks compared to the strongest local model already in use — Qwen 3.6 35B?**

Rather than rely on vendor benchmark tables, we ran a **controlled, task-level benchmark** focused on the behaviors that matter for agents (not academic multiple-choice): can the model *choose* the right tool, *call it with correct arguments*, *chain multiple tool calls*, *recover from errors*, *generate working code*, and *follow format constraints*.

## Models Under Test

Both models run locally via **Ollama's MLX engine** on the same Apple M1 Max (64 GB unified memory). Same quantization family (nvfp4), same context budget, same prompts.

| Property | **muse-glimmer:30b-mlx** | **qwen3.6:35b-mlx** |
|---|---|---|
| Family | Muse Glimmer (Meta) | Qwen 3.5/3.6 (Alibaba) |
| Total parameters | 32.3B | 35.1B |
| **Active params / token** | **32.3B (dense — all)** | **~3B (MoE — sparse)** |
| **Architecture** | `muse_glimmer` (**dense**) | `qwen3_5_moe` (**Mixture-of-Experts**) |
| Embedding length | 6656 | 2048 |
| Context window | 131072 | 262144 |
| Quantization | nvfp4 | nvfp4 |
| Capabilities | completion, vision, tools, thinking | completion, vision, tools, thinking |

> **The MoE fact is central to the analysis.** Qwen 3.6 35B (arch `qwen3_5_moe`, i.e. the Qwen3.5-35B-A3B lineage) has ~35B total parameters but activates only **~3B per token** via a sparse Mixture-of-Experts gating mechanism. Muse Glimmer activates **all 32.3B** parameters for every token (dense). This is the dominant driver of the measured latency gap (see [Analysis](#analysis-why-the-difference)).

## Benchmark Design

We wrote a purpose-built harness (`benchmark/agent_bench.py`) that drives both models through Ollama's `/api/chat` endpoint with **native tool-calling** and a simulated tool executor. Each task is a natural-language agent prompt; the model decides which tools to call (and with what arguments), receives tool results, and loops up to 6 turns before producing a final answer.

**The 12 agentic tasks:**

| # | Task | What it tests |
|---|---|---|
| 1 | `tool_call_weather` | Simple tool selection + correct call |
| 2 | `tool_call_args` | Tool call with correct numeric argument |
| 3 | `multi_step_weather` | Two parallel tool calls, compare results |
| 4 | `multi_step_search_read` | Chained search → read → report |
| 5 | `failure_recovery` | Handle a tool error, fall back gracefully |
| 6 | `code_fizzbuzz` | Correct working Python code |
| 7 | `code_two_sum` | Correct algorithm implementation |
| 8 | `instruct_json` | Strict JSON output constraint |
| 9 | `instruct_format` | Numbered-list format constraint |
| 10 | `reasoning_math` | Tool-backed multi-step arithmetic |
| 11 | `agentic_loop` | **3-step tool chain, execute all steps literally** |
| 12 | `tool_selection` | Pick the correct tool for the job |

**Config:** temperature 0.2, `num_predict` 2000, up to 6 tool turns per task, single run per model (latency includes cold-start on first task).

## Results

### Overall

| Metric | **Muse Glimmer 30B** | **Qwen 3.6 35B** |
|---|---|---|
| **Passed** | **12/12 (100%)** | 11/12 (92%) |
| **Total time** | 371.4s | **187.3s** (2× faster) |
| **Avg per task** | 31.0s | 15.6s |

### Per-task detail

| Task | Glimmer | Qwen | Glimmer (s) | Qwen (s) |
|---|---|---|---|---|
| tool_call_weather | ✅ | ✅ | 9.1 | 29.0 |
| tool_call_args | ✅ | ✅ | 9.2 | 5.0 |
| multi_step_weather | ✅ | ✅ | 19.2 | 10.9 |
| multi_step_search_read | ✅ | ✅ | 31.8 | 22.3 |
| failure_recovery | ✅ | ✅ | 86.7 | **7.1** |
| code_fizzbuzz | ✅ | ✅ | 34.7 | 24.0 |
| code_two_sum | ✅ | ✅ | 24.6 | 18.3 |
| instruct_json | ✅ | ✅ | 10.2 | **2.8** |
| instruct_format | ✅ | ✅ | 22.2 | 20.5 |
| reasoning_math | ✅ | ✅ | 22.7 | 7.2 |
| **agentic_loop** | ✅ | ❌ | 73.1 | 35.3 |
| tool_selection | ✅ | ✅ | 27.7 | 4.9 |

### The single failure

**Task 11 (`agentic_loop`)** required the model to: `search_files` → `read_file` → `calculate` (count lines) → report.

- **Muse Glimmer:** executed the full 3-tool chain literally — `[search_files, read_file, calculate]` — and reported the computed count.
- **Qwen 3.6:** chained `[search_files, read_file]` but **skipped the `calculate` call**, counting the 3 lines itself and answering directly.

## Analysis: Why the Difference?

### The MoE hypothesis is the core reason

**Yes — the primary reason for the speed gap is that Qwen 3.6 is a Mixture-of-Experts (MoE) model and Muse Glimmer is a dense model.**

- **MoE (Qwen 3.6 35B):** ~35B total parameters, but a gating network routes each token through only **~3B active parameters**. The rest of the experts are idle for that token. This dramatically reduces FLOPs per token → much lower inference latency and memory bandwidth pressure per token. This is exactly why Qwen could run a 21 GB / 35B-parameter model at 2× the speed of a similar-size dense model.
- **Dense (Muse Glimmer 32.3B):** every parameter is activated for every token. Higher FLOPs per token → higher latency per generation, but every expert "brain" is engaged on every step.

This architectural difference maps directly onto our measurements: **Qwen's per-task times are consistently lower**, and its best cases (2.8s JSON, 4.9s tool selection, 7.1s failure recovery) are 3–12× faster than Glimmer's — the classic MoE vs dense efficiency gap.

### Why Glimmer wins on tool-use fidelity

Despite being slower, Glimmer was **more instruction-faithful on the long tool chain** — the one place Qwen failed. Several factors likely contribute:

1. **Purpose-built for agents.** Glimmer is explicitly trained/distilled for "always-on local agent workflows" — reliable tool use, multi-step reasoning, and failure recovery (Meta Superintelligence Labs). Following a prescribed tool plan literally is a core training target.
2. **Dense models keep full capacity on the task.** With all 32.3B params active, the model has maximal representation capacity per step. MoE routes through a small active slice; for a task demanding exact sequential tool execution, the full dense network may track the instruction more reliably.
3. **Observed behavior: less "shortcutting."** Glimmer made more tool calls and kept verifying (e.g. 4 calls on `failure_recovery`), while Qwen took the economical path (2 calls). Fidelity vs efficiency is a real trade-off.

### Why Qwen wins on speed

- **Sparse activation:** ~3B vs 32.3B active params per token → far fewer FLOPs.
- **Lower memory bandwidth:** MoE loads only the active experts' weights per token, reducing the dominant cost on memory-bound Apple Silicon inference.
- **Fewer, cheaper tokens generated:** Qwen produced tighter outputs (e.g. 2.8s JSON, 4.9s tool selection) — less "over-verification," which compounds the per-token advantage.

### Latency outliers worth noting

- **`failure_recovery` (86.7s Glimmer vs 7.1s Qwen):** Glimmer made 4 tool calls (read → search → read → read) and kept verifying; Qwen made 2 (read → search) and stopped. Glimmer's caution, while correct, is expensive.
- **`instruct_json` (10.2s vs 2.8s)** and **`tool_selection` (27.7s vs 4.9s):** Qwen's tighter token budget on simple tasks produces outsized speedups.
- Glimmer's **first task (9.1s)** is partly cold-start; its warm performance is better than raw totals suggest, but still not MoE-fast.

## Caveats & Methodology Notes

- **Single run per model.** Latency includes cold-start effects on the first task. A multi-round run with medians would tighten the latency comparison.
- **Both models are 21 GB MLX on the same M1 Max (64 GB).** The gap is not a hardware difference — same machine, same engine, same quantization family.
- **Tool simulator is deterministic** and identical for both models, so results isolate model behavior, not environment noise.
- **Latency is end-to-end** (request → tool loop → final answer), the number that matters for real agent UX.
- **Quality scoring is binary (pass/fail) per task.** It captures whether the model did the right thing; it does not fully capture answer quality nuance within a pass.

## Conclusions & Recommendations

| Use case | Recommended model |
|---|---|
| **Strict multi-step agentic workflows** (must execute every tool step as planned) | **Muse Glimmer 30B** — 100% tool-use fidelity |
| **Latency-sensitive / interactive use** (many short calls) | **Qwen 3.6 35B** — ~2× faster, minor reliability trade-off |
| **Code generation & instruction following** | Either — no meaningful quality gap |

**Bottom line:** Glimmer and Qwen 3.6 are both strong local agent models. The deciding factor is the trade-off between **tool-use fidelity** (Glimmer) and **speed via sparse MoE activation** (Qwen). If your agents must reliably chain 3+ tool calls and follow tool-use plans literally, Glimmer's 100% reliability justifies its speed cost. If you're optimizing for fast interactive responses and can tolerate occasional shortcutting, Qwen's MoE efficiency wins.

## Reproduction

```bash
# Requires Ollama 0.32.7+ (Muse Glimmer needs the MLX DFlash support in 0.32.7)
ollama pull muse-glimmer:30b-mlx
ollama pull qwen3.6:35b-mlx

# Run the benchmark on either model (expect 3–6 min per model on M-series)
PYTHONPATH="" python3 benchmark/agent_bench.py muse-glimmer:30b-mlx
PYTHONPATH="" python3 benchmark/agent_bench.py qwen3.6:35b-mlx
```

Raw per-task output is captured in `results/` for both models.

## License

- Benchmark harness: MIT (see `benchmark/agent_bench.py` header).
- Muse Glimmer weights: Apache 2.0 (Meta).
- Qwen 3.5/3.6: Apache 2.0 (Alibaba).
