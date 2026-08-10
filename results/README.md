# Raw results

Per-task raw output (JSON) from the benchmark runs:

- `muse_glimmer_30b.json` — Muse Glimmer 30B (dense): 12/12 passed, 371.4s total
- `qwen3.6_35b.json` — Qwen 3.6 35B (MoE): 11/12 passed, 187.3s total

Each file contains the model name, pass/total, total wall time, and per-task
results (pass/fail, elapsed seconds, tool calls made, and a sample of the
final answer text). Reproduce with `benchmark/agent_bench.py`.
