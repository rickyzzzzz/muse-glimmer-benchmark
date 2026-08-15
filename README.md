# Muse Glimmer 30B vs Qwen 3.6 35B vs Qwen 3.8 27B — Local Agent-Task Benchmark

> **A controlled comparison of Meta's Muse Glimmer 30B, Alibaba's Qwen 3.6 35B, and Qwen 3.8 27B on real agentic workloads** — tool calling, multi-step tool chains, failure recovery, code generation, and instruction following — all running locally on the same Apple M1 Max (64 GB) via Ollama's MLX engine.

---

## TLDR

- **Muse Glimmer 30B passed 12/12 agent tasks (100%)**; **Qwen 3.6 35B passed 11/12 (92%)**; **Qwen 3.8 27B passed 12/12 (100%)**.
- **Qwen 3.8 is the fastest of the three** (180.7s total vs 187.3s for Qwen 3.6 and 371.4s for Glimmer) — and it fixed the one task Qwen 3.6 failed.
- **The decisive difference (Qwen 3.6 vs 3.8):** on a 3-step tool chain (search → read → calculate), Qwen 3.6 shortcut past the `calculate` step and answered itself; **Qwen 3.8 executed every step as instructed** — matching Glimmer's tool-use fidelity.
- **Why the speed gap — the key architectural factor:** Glimmer is a **dense** model (~32.3B params, all active per token); Qwen 3.6 35B is a **Mixture-of-Experts (MoE)** model (~35B total, only **~3B active per token**); **Qwen 3.8 27B is dense** (~27.8B params, all active) — yet still ~2× faster than Glimmer.
- **Recommendation:** Qwen 3.8 27B is now the best all-round local agent model of the three — Glimmer-level tool-use fidelity at Qwen-level speed.

| Metric | **Muse Glimmer 30B** | **Qwen 3.6 35B** | **Qwen 3.8 27B** |
|---|---|---|---|
| **Passed** | **12/12 (100%)** | 11/12 (92%) | **12/12 (100%)** |
| **Total time** | 371.4s | 187.3s | **180.7s** (fastest) |
| **Avg per task** | 31.0s | 15.6s | **15.1s** |
| **Architecture** | Dense | MoE | **Dense** |

![Benchmark infographic](infographic/infographic.png)

---

## Table of Contents

1. [Background & Motivation](#background--motivation)
2. [Models Under Test](#models-under-test)
3. [Benchmark Design](#benchmark-design)
4. [Results](#results)
5. [Analysis: Why the Difference?](#analysis-why-the-difference)
6. [Caveats & Methodology Notes](#caveats--methodology-notes)
7. [Conclusions & Recommendations](#conclusions--recommendations)
8. [Reproduction](#reproduction)
9. [License](#license)

---

## Background & Motivation

Muse Glimmer is Meta's open-weight (Apache 2.0) 30B model, built by Meta Superintelligence Labs specifically for **always-on local agent workflows** — reliable tool use, multi-step reasoning, failure recovery, and long-horizon tasks on consumer hardware.

We wanted to answer a practical question: **how does it actually perform on agentic tasks compared to the strongest local models in use — Qwen 3.6 35B and the newer Qwen 3.8 27B?**

Rather than rely on vendor benchmark tables, we ran a **controlled, task-level benchmark** focused on the behaviors that matter for agents (not academic multiple-choice): can the model *choose* the right tool, *call it with correct arguments*, *chain multiple tool calls*, *recover from errors*, *generate working code*, and *follow format constraints*.

## Models Under Test

All models run locally via **Ollama's MLX engine** on the same Apple M1 Max (64 GB unified memory). Same quantization family (nvfp4), same context budget, same prompts.

| Property | **muse-glimmer:30b-mlx** | **qwen3.6:35b-mlx** | **qwen3.8:27b-mlx** |
|---|---|---|---|
| Family | Muse Glimmer (Meta) | Qwen 3.5/3.6 (Alibaba) | Qwen 3.8 (Alibaba) |
| Total parameters | 32.3B | 35.1B | 27.8B |
| **Active params / token** | **32.3B (dense — all)** | **~3B (MoE — sparse)** | **27.8B (dense — all)** |
| **Architecture** | `muse_glimmer` (**dense**) | `qwen3_5_moe` (**Mixture-of-Experts**) | `qwen3_5` (**dense**) |
| Embedding length | 6656 | 2048 | 5120 |
| Context window | 131072 | 262144 | 262144 |
| Quantization | nvfp4 | nvfp4 | nvfp4 |
| Capabilities | completion, vision, tools, thinking | completion, vision, tools, thinking | completion, vision, tools, thinking |

> **The architecture facts are central to the analysis.** Qwen 3.6 35B (arch `qwen3_5_moe`, i.e. the Qwen3.5-35B-A3B lineage) has ~35B total parameters but activates only **~3B per token** via a sparse Mixture-of-Experts gating mechanism. Muse Glimmer activates **all 32.3B** parameters for every token (dense). **Qwen 3.8 27B is dense** (arch `qwen3_5`, no expert routing) — all 27.8B params active per token — yet still runs ~2× faster than Glimmer, likely due to its smaller parameter count and a more efficient MLX implementation.

## Benchmark Design

We wrote a purpose-built harness (`benchmark/agent_bench.py`) that drives all models through Ollama's `/api/chat` endpoint with **native tool-calling** and a simulated tool executor. Each task is a natural-language agent prompt; the model decides which tools to call (and with what arguments), receives tool results, and loops up to 6 turns before producing a final answer.

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

| Metric | **Muse Glimmer 30B** | **Qwen 3.6 35B** | **Qwen 3.8 27B** |
|---|---|---|---|
| **Passed** | **12/12 (100%)** | 11/12 (92%) | **12/12 (100%)** |
| **Total time** | 371.4s | 187.3s | **180.7s** (fastest) |
| **Avg per task** | 31.0s | 15.6s | **15.1s** |

### Per-task detail

| Task | Glimmer | Q3.6 | Q3.8 | Glimmer (s) | Q3.6 (s) | Q3.8 (s) |
|---|---|---|---|---|---|---|
| tool_call_weather | ✅ | ✅ | ✅ | 9.1 | 29.0 | 13.5 |
| tool_call_args | ✅ | ✅ | ✅ | 9.2 | 5.0 | 11.9 |
| multi_step_weather | ✅ | ✅ | ✅ | 19.2 | 10.9 | 12.7 |
| multi_step_search_read | ✅ | ✅ | ✅ | 31.8 | 22.3 | **17.4** |
| failure_recovery | ✅ | ✅ | ✅ | 86.7 | **7.1** | 35.9 |
| code_fizzbuzz | ✅ | ✅ | ✅ | 34.7 | 24.0 | **15.4** |
| code_two_sum | ✅ | ✅ | ✅ | 24.6 | 18.3 | **5.3** |
| instruct_json | ✅ | ✅ | ✅ | 10.2 | **2.8** | 3.5 |
| instruct_format | ✅ | ✅ | ✅ | 22.2 | 20.5 | **8.5** |
| reasoning_math | ✅ | ✅ | ✅ | 22.7 | 7.2 | 10.3 |
| **agentic_loop** | ✅ | ❌ | ✅ | 73.1 | 35.3 | **32.6** |
| tool_selection | ✅ | ✅ | ✅ | 27.7 | 4.9 | 13.6 |

### The single failure (Qwen 3.6 only)

**Task 11 (`agentic_loop`)** required the model to: `search_files` → `read_file` → `calculate` (count lines) → report.

- **Muse Glimmer:** executed the full 3-tool chain literally — `[search_files, read_file, calculate]` — and reported the computed count.
- **Qwen 3.6:** chained `[search_files, read_file]` but **skipped the `calculate` call**, counting the 3 lines itself and answering directly.
- **Qwen 3.8:** executed the full 3-tool chain — `[search_files, read_file, calculate]` — and reported the computed count, **matching Glimmer's tool-use fidelity**.

## Analysis: Why the Difference?

### The architecture story (3 models, 2 architectures)

- **MoE (Qwen 3.6 35B):** ~35B total parameters, but a gating network routes each token through only **~3B active parameters**. This dramatically reduces FLOPs per token → much lower inference latency. This is why Qwen 3.6 ran at 2× the speed of the similar-size dense Glimmer.
- **Dense (Muse Glimmer 32.3B):** every parameter is activated for every token. Higher FLOPs per token → higher latency per generation, but every "brain" is engaged on every step.
- **Dense (Qwen 3.8 27B):** all 27.8B params active per token, yet **still the fastest of the three** (180.7s). The smaller parameter count (27.8B vs 32.3B) plus a more efficient MLX implementation more than compensates for the dense-vs-MoE gap.

### Why Qwen 3.8 wins on tool-use fidelity (vs Qwen 3.6)

Qwen 3.8 fixed the one failure Qwen 3.6 had — it now executes 3-step tool chains literally, matching Glimmer. This suggests the newer model generation improved instruction-following for sequential tool plans, closing the fidelity gap that previously favored Glimmer.

### Why Qwen 3.8 wins on speed

- **Smaller dense model:** 27.8B params vs Glimmer's 32.3B — fewer FLOPs per token.
- **Efficient MLX implementation:** the `qwen3_5` dense architecture runs well on Apple Silicon.
- **Tighter outputs:** Qwen 3.8 produced concise answers (e.g. 5.3s two_sum, 8.5s format) — less "over-verification" than Glimmer.

### Latency outliers worth noting

- **`failure_recovery` (86.7s Glimmer vs 7.1s Q3.6 vs 35.9s Q3.8):** Glimmer made 4 tool calls (read → search → read → read) and kept verifying; Qwen 3.6 made 2 (read → search) and stopped; Qwen 3.8 made 4 (read → search → read → read) like Glimmer but faster.
- **`code_two_sum` (24.6s Glimmer vs 18.3s Q3.6 vs 5.3s Q3.8):** Qwen 3.8's tightest win — 4.6× faster than Glimmer.
- Glimmer's **first task (9.1s)** is partly cold-start; its warm performance is better than raw totals suggest, but still not Qwen-fast.

## Caveats & Methodology Notes

- **Single run per model.** Latency includes cold-start effects on the first task. A multi-round run with medians would tighten the latency comparison.
- **All models are MLX on the same M1 Max (64 GB).** The gap is not a hardware difference — same machine, same engine, same quantization family.
- **Tool simulator is deterministic** and identical for all models, so results isolate model behavior, not environment noise.
- **Latency is end-to-end** (request → tool loop → final answer), the number that matters for real agent UX.
- **Quality scoring is binary (pass/fail) per task.** It captures whether the model did the right thing; it does not fully capture answer quality nuance within a pass.

## Conclusions & Recommendations

| Use case | Recommended model |
|---|---|
| **Strict multi-step agentic workflows** (must execute every tool step as planned) | **Qwen 3.8 27B** or **Muse Glimmer 30B** — both 100% tool-use fidelity; Qwen 3.8 is 2× faster |
| **Latency-sensitive / interactive use** (many short calls) | **Qwen 3.8 27B** — fastest overall |
| **Code generation & instruction following** | **Qwen 3.8 27B** — fastest on both code tasks |
| **Legacy Qwen 3.6 workloads** | **Migrate to Qwen 3.8** — same speed class, better fidelity |

**Bottom line:** Qwen 3.8 27B is the best all-round local agent model of the three — it matches Glimmer's 100% tool-use fidelity while being ~2× faster, and it fixes the one reliability gap Qwen 3.6 had. Glimmer remains a strong choice where its specific training for always-on agent workflows matters, but for most local agent workloads Qwen 3.8 is now the pick.

## Reproduction

```bash
# Requires Ollama 0.32.7+ (Muse Glimmer needs the MLX DFlash support in 0.32.7)
ollama pull muse-glimmer:30b-mlx
ollama pull qwen3.8:27b-mlx

# Run the benchmark on any model (expect 3–6 min per model on M-series)
PYTHONPATH="" python3 benchmark/agent_bench.py muse-glimmer:30b-mlx
PYTHONPATH="" python3 benchmark/agent_bench.py qwen3.8:27b-mlx
```

Raw per-task output is captured in `results/` for all models.

## License

- Benchmark harness: MIT (see `benchmark/agent_bench.py` header).
- Muse Glimmer weights: Apache 2.0 (Meta).
- Qwen 3.5/3.6/3.8: Apache 2.0 (Alibaba).
