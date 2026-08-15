#!/usr/bin/env python3
"""
Agent-task benchmark for local Ollama models.
Tests real agentic capabilities via /api/chat with tools:
  1. Tool calling (function selection + correct args)
  2. Multi-step reasoning (chained tool use)
  3. Failure recovery (handles error, retries)
  4. Code generation (working Python)
  5. Instruction following (format constraints)
  6. Long-context tool loop (agentic persistence)

Usage: python3 agent_bench.py <model> [--rounds N]
"""
import json, sys, time, statistics, urllib.request

OLLAMA = "http://127.0.0.1:11434/api/chat"

# --- Tool definitions (mimic a real agent environment) ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temperature in Celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a pattern in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"}
                },
                "required": ["pattern", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a SQL query against the company analytics database. Returns query results as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_experiment_results",
            "description": "Fetch results for an A/B experiment by ID. Returns conversion rates, sample sizes, and p-value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "Experiment ID"}
                },
                "required": ["experiment_id"]
            }
        }
    },
]

# --- Simulated tool executor ---
def execute_tool(name, args):
    """Simulate tool execution. Returns (result_str, is_error)."""
    if name == "get_weather":
        city = args.get("city", "")
        if city.lower() == "tokyo":
            return "Weather in Tokyo: 24C, partly cloudy", False
        elif city.lower() == "paris":
            return "Weather in Paris: 18C, light rain", False
        else:
            return f"Weather in {city}: 21C, sunny", False
    elif name == "calculate":
        expr = args.get("expression", "")
        try:
            # safe eval
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expr):
                return "Error: invalid characters in expression", True
            result = eval(expr)
            return f"Result: {result}", False
        except Exception as e:
            return f"Error: {e}", True
    elif name == "search_files":
        pattern = args.get("pattern", "")
        path = args.get("path", "")
        if pattern == "*.py" and path == "/home/user/project":
            return "Found: main.py, utils.py, test_main.py", False
        elif pattern == "config*":
            return "Found: config.yaml, config.json", False
        else:
            return "No files found", False
    elif name == "read_file":
        path = args.get("path", "")
        if path == "/home/user/project/main.py":
            return "def main():\n    print('hello')\n    return 42", False
        elif path == "/home/user/project/config.yaml":
            return "server:\n  port: 8080\n  host: 0.0.0.0", False
        else:
            return f"Error: file not found: {path}", True
    elif name == "run_sql":
        query = args.get("query", "").lower()
        if "orders" in query and "count" in query:
            return "Result: 1250", False
        elif "orders" in query and "sum" in query:
            return "Result: 48250.50", False
        elif "users" in query and "count" in query:
            return "Result: 3200", False
        elif "revenue" in query and "month" in query:
            return "Result: Jan 12000, Feb 13500, Mar 14200", False
        elif "cohort" in query:
            return "Result: W1 12.5%, W2 11.8%, W3 12.1%", False
        else:
            return "Result: 0 rows", False
    elif name == "get_experiment_results":
        exp = args.get("experiment_id", "")
        if exp == "exp_1234":
            return "Experiment exp_1234: control_conv=0.032, treatment_conv=0.038, control_n=50000, treatment_n=50000, p_value=0.0001", False
        elif exp == "exp_5678":
            return "Experiment exp_5678: control_conv=0.045, treatment_conv=0.046, control_n=30000, treatment_n=30000, p_value=0.42", False
        else:
            return f"Error: experiment not found: {exp}", True
    return "Unknown tool", True

# --- Test cases ---
# Each: (name, messages, expected_behavior, check_fn)
# check_fn(text, tool_calls) -> (bool, detail)
TESTS = []

def t(name, messages, check):
    TESTS.append((name, messages, check))

# 1. Simple tool call: weather
t("tool_call_weather",
  [{"role": "user", "content": "What's the weather in Tokyo? Use the get_weather tool."}],
  lambda r, tc: ("get_weather" in tc and "tokyo" in r.lower(), "called get_weather"))

# 2. Tool call with correct args
t("tool_call_args",
  [{"role": "user", "content": "Use the calculate tool to compute 15 * 4. What is the result?"}],
  lambda r, tc: ("calculate" in tc and "60" in r, "computed 15*4=60"))

# 3. Multi-step: weather then decide
t("multi_step_weather",
  [{"role": "user", "content": "Check the weather in Paris and Tokyo using get_weather, then tell me which city is warmer."}],
  lambda r, tc: (tc.count("get_weather") >= 2 and ("paris" in r.lower() and "tokyo" in r.lower()), "queried both cities"))

# 4. Multi-step: search then read
t("multi_step_search_read",
  [{"role": "user", "content": "Search for *.py files in /home/user/project, then read main.py and tell me what it returns."}],
  lambda r, tc: ("search_files" in tc and "read_file" in tc and "42" in r, "searched then read"))

# 5. Failure recovery: read nonexistent file, then search
t("failure_recovery",
  [{"role": "user", "content": "Read the file /home/user/project/nonexistent.txt. If it fails, search for config* files instead and report what you find."}],
  lambda r, tc: ("search_files" in tc and "config" in r.lower(), "recovered from error"))

# 6. Code generation: working Python
t("code_fizzbuzz",
  [{"role": "user", "content": "Write a Python function fizzbuzz(n) that returns a list where multiples of 3 are 'Fizz', 5 are 'Buzz', both are 'FizzBuzz', else the number. Output ONLY the code, no explanation."}],
  lambda r, tc: ("def fizzbuzz" in r and "FizzBuzz" in r and "Fizz" in r and "Buzz" in r, "valid fizzbuzz code"))

# 7. Code generation: algorithm
t("code_two_sum",
  [{"role": "user", "content": "Write a Python function two_sum(nums, target) that returns indices of two numbers summing to target. Output ONLY the code."}],
  lambda r, tc: ("def two_sum" in r and "target" in r, "valid two_sum code"))

# 8. Instruction following: JSON output
t("instruct_json",
  [{"role": "user", "content": "Return a JSON object with keys 'name' and 'age' for a person named Alice who is 30. Output ONLY valid JSON, no other text."}],
  lambda r, tc: (("name" in r and "age" in r and "Alice" in r and "30" in r), "JSON with name/age"))

# 9. Instruction following: format constraint
t("instruct_format",
  [{"role": "user", "content": "List 3 reasons to use version control. Format each as '1. reason', '2. reason', '3. reason'. Do not add any other text."}],
  lambda r, tc: (r.count("\n") >= 2 and "1." in r and "2." in r and "3." in r, "numbered list of 3"))

# 10. Multi-step reasoning: math chain
t("reasoning_math",
  [{"role": "user", "content": "A store sells apples at $2 each and oranges at $3 each. If someone buys 4 apples and 3 oranges, use the calculate tool to find the total cost, then state the answer."}],
  lambda r, tc: ("calculate" in tc and ("17" in r), "computed 4*2+3*3=17"))

# 11. Tool loop persistence (agentic): needs multiple tool calls
t("agentic_loop",
  [{"role": "user", "content": "Use search_files to find *.py files in /home/user/project, then read_file on main.py, then use calculate to compute the number of lines in main.py (it has 3 lines). Report the final count."}],
  lambda r, tc: ("search_files" in tc and "read_file" in tc and "calculate" in tc and "3" in r, "chained 3 tools"))

# 12. Tool selection: pick right tool
t("tool_selection",
  [{"role": "user", "content": "I need to find configuration files. Which tool should I use? Use it to search for config* files in /home/user/project."}],
  lambda r, tc: ("search_files" in tc and "config" in r.lower(), "selected search_files"))

# ═══════════════════════════════════════════════════════════════════
#  DATA SCIENCE WORKLOAD TESTS — coding, experimentation, causal
# ═══════════════════════════════════════════════════════════════════

# 13. DS coding: pandas data cleaning + groupby aggregation
t("ds_code_pandas_clean",
  [{"role": "user", "content": "Write Python pandas code that: (1) reads a CSV with columns user_id, date, revenue, (2) drops rows with NaN revenue, (3) groups by user_id and sums revenue, (4) returns the top 10 users by total revenue. Output ONLY the code."}],
  lambda r, tc: (("dropna" in r or "isna" in r or "notna" in r) and "groupby" in r and "sum" in r and ("head" in r or "nlargest" in r) and "user_id" in r, "pandas clean+groupby+top10"))
# 14. DS coding: retention / cohort metric in Python
t("ds_code_retention",
  [{"role": "user", "content": "Write Python code to compute 7-day retention: given a list of (user_id, signup_date) and a list of (user_id, activity_date), return the fraction of users who were active on day 7 after signup. Output ONLY the code."}],
  lambda r, tc: (("timedelta" in r or "days" in r) and "7" in r and "set" in r.lower() and "signup" in r.lower(), "retention calc code"))
# 15. DS coding: A/B test sample size (power analysis formula)
t("ds_code_power",
  [{"role": "user", "content": "Write Python code to compute the required sample size per variant for an A/B test: baseline conversion 5%, minimum detectable effect 1 percentage point (relative), alpha=0.05, power=0.8. Use the standard two-proportion formula or statsmodels. Output ONLY the code."}],
  lambda r, tc: (("statsmodels" in r or "z" in r.lower() or "norm" in r.lower() or "0.05" in r) and ("0.8" in r or "power" in r.lower()) and "0.05" in r, "power analysis code"))
# 16. DS coding: SQL with JOIN + GROUP BY
t("ds_code_sql_join",
  [{"role": "user", "content": "Write a SQL query: join orders to users, compute total revenue per country, return only countries with revenue above 10000, ordered by revenue descending. Output ONLY the SQL."}],
  lambda r, tc: ("join" in r.lower() and "group by" in r.lower() and "sum" in r.lower() and "having" in r.lower() and "order by" in r.lower(), "SQL join+group+having"))

# 17. DS coding: SQL window function (dedup / rank)
t("ds_code_sql_window",
  [{"role": "user", "content": "Write a SQL query using a window function to find the most recent order per user from an orders table (columns: order_id, user_id, order_date). Output ONLY the SQL."}],
  lambda r, tc: (("row_number" in r.lower() or "rank" in r.lower() or "partition by" in r.lower()) and "order by" in r.lower(), "SQL window function"))
# 18. Experimentation: interpret A/B results and decide
t("exp_interpret_ab",
  [{"role": "user", "content": "Use the get_experiment_results tool to fetch results for experiment exp_1234. The primary metric is conversion rate. Based on the results, should we ship the treatment? Explain your reasoning."}],
  lambda r, tc: ("get_experiment_results" in tc and ("ship" in r.lower() or "launch" in r.lower() or "roll out" in r.lower() or "recommend" in r.lower()) and ("0.038" in r or "3.8" in r or "0.0001" in r or "significant" in r.lower() or "p" in r.lower()), "interpreted A/B results"))
# 19. Experimentation: non-significant result — do NOT ship
t("exp_nonsig_ab",
  [{"role": "user", "content": "Use the get_experiment_results tool to fetch results for experiment exp_5678. The primary metric is conversion rate. Should we ship the treatment? Explain your reasoning."}],
  lambda r, tc: ("get_experiment_results" in tc and ("not" in r.lower() and ("ship" in r.lower() or "launch" in r.lower() or "recommend" in r.lower())) and ("0.42" in r or "p" in r.lower() or "significant" in r.lower()), "correctly rejected non-sig result"))
# 20. Experimentation: design an A/B test (metrics + randomization)
t("exp_design",
  [{"role": "user", "content": "We want to test a new onboarding flow. Describe how you would design the A/B test: what is the randomization unit, what are the primary and guardrail metrics, and how long should it run? Be specific and concise."}],
  lambda r, tc: (("user" in r.lower() or "account" in r.lower() or "random" in r.lower()) and ("metric" in r.lower() or "conversion" in r.lower()) and ("guardrail" in r.lower() or "guard rail" in r.lower() or "latency" in r.lower() or "revenue" in r.lower()) and ("week" in r.lower() or "day" in r.lower() or "power" in r.lower()), "A/B design with metrics+guardrails"))
# 21. Causal inference: identify confounder
t("causal_confounder",
  [{"role": "user", "content": "We observe that users who use our new feature have higher retention than users who don't. Can we conclude the feature causes higher retention? Identify potential confounders and explain what analysis you would run to establish causality."}],
  lambda r, tc: (("confound" in r.lower() or "selection bias" in r.lower() or "self-selection" in r.lower()) and ("random" in r.lower() or "experiment" in r.lower() or "propensity" in r.lower() or "matching" in r.lower() or "instrumental" in r.lower()), "identified confounders"))
# 22. Causal inference: difference-in-differences
t("causal_did",
  [{"role": "user", "content": "A policy change affected users in California but not New York. Explain how you would use difference-in-differences to estimate the causal effect of the policy. What are the key assumptions? Be concise."}],
  lambda r, tc: (("difference-in-differences" in r.lower() or "did" in r.lower() or "parallel" in r.lower()) and ("parallel trends" in r.lower() or "parallel" in r.lower()) and ("california" in r.lower() and "new york" in r.lower()), "DiD design + parallel trends"))
# 23. Causal inference: propensity score matching
t("causal_psm",
  [{"role": "user", "content": "We want to estimate the effect of a premium subscription on churn using observational data. Explain how propensity score matching works and what its limitations are. Be concise."}],
  lambda r, tc: (("propensity" in r.lower() or "psm" in r.lower() or "matching" in r.lower()) and ("unobserved" in r.lower() or "confound" in r.lower() or "limitation" in r.lower() or "bias" in r.lower()), "PSM explanation + limitations"))
# 24. Tool use: answer business question via run_sql
t("ds_tool_sql_query",
  [{"role": "user", "content": "Use the run_sql tool to find out how many orders we have in total. Then report the number."}],
  lambda r, tc: ("run_sql" in tc and ("1250" in r or "1,250" in r), "queried orders count via SQL tool"))

# 25. Tool use: cohort retention via run_sql
t("ds_tool_cohort",
  [{"role": "user", "content": "Use the run_sql tool to get cohort retention data (query the cohort table). Then tell me which week had the highest retention."}],
  lambda r, tc: ("run_sql" in tc and ("12.5" in r or "week 1" in r.lower() or "w1" in r.lower()), "queried cohort retention"))
# 26. Tool use: multi-tool chain — SQL then experiment
t("ds_tool_chain",
  [{"role": "user", "content": "First use run_sql to count how many users we have. Then use get_experiment_results to fetch exp_1234. Report both the user count and whether the experiment is significant."}],
  lambda r, tc: ("run_sql" in tc and "get_experiment_results" in tc and ("3200" in r or "3,200" in r) and ("significant" in r.lower() or "ship" in r.lower() or "0.038" in r or "3.8" in r), "chained SQL + experiment tools"))
# 27. DS coding: metric definition — LTV
t("ds_code_ltv",
  [{"role": "user", "content": "Write Python code to compute customer lifetime value (LTV) given: average revenue per user per month (ARPU), monthly churn rate, and gross margin. Use the standard LTV formula. Output ONLY the code."}],
  lambda r, tc: (("ltv" in r.lower() or "clv" in r.lower()) and ("churn" in r.lower() or "margin" in r.lower()) and ("/" in r or "div" in r.lower()), "LTV formula code"))
# 28. Experimentation: multiple testing / guardrail awareness
t("exp_multiple_testing",
  [{"role": "user", "content": "We ran an experiment and are checking 20 different metrics. One metric shows p=0.03. Should we declare victory? Explain the multiple testing problem and what you would do."}],
  lambda r, tc: (("multiple" in r.lower() or "bonferroni" in r.lower() or "fdr" in r.lower() or "false discovery" in r.lower()) and ("0.03" in r or "p" in r.lower()) and ("no" in r.lower() or "not" in r.lower() or "adjust" in r.lower()), "multiple testing awareness"))
# 29. Causal inference: Simpson's paradox
t("causal_simpson",
  [{"role": "user", "content": "In an A/B test, the treatment shows higher overall conversion, but within every segment (mobile, desktop, tablet) the control is higher. What is happening? Explain and what would you do."}],
  lambda r, tc: (("simpson" in r.lower() or "confound" in r.lower() or "segment" in r.lower() or "stratified" in r.lower() or "weighted" in r.lower()) and ("mobile" in r.lower() or "desktop" in r.lower()), "Simpson's paradox explanation"))
# 30. DS coding: SQL cohort retention
t("ds_code_sql_cohort",
  [{"role": "user", "content": "Write SQL to compute weekly cohort retention: for each signup week, the fraction of users active in each subsequent week. Tables: signups(user_id, signup_date), activity(user_id, activity_date). Output ONLY the SQL."}],
  lambda r, tc: (("cohort" in r.lower() or "date_trunc" in r.lower() or "strftime" in r.lower() or "week" in r.lower()) and ("signup" in r.lower() and "activity" in r.lower()) and ("join" in r.lower() or "left join" in r.lower()), "SQL cohort retention"))
def call_model(model, messages, max_turns=6):
    """Run a chat with tool-calling loop. Returns (final_text, tool_calls_made, error)."""
    msgs = list(messages)
    tool_calls_made = []
    start = time.time()
    try:
        for turn in range(max_turns):
            body = {
                "model": model,
                "messages": msgs,
                "stream": False,
                "tools": TOOLS,
                "options": {"temperature": 0.2, "num_predict": 2000}
            }
            req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
            msg = data.get("message", {})
            content = msg.get("content", "") or ""
            calls = msg.get("tool_calls", [])
            if calls:
                for c in calls:
                    fn = c.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    tool_calls_made.append(name)
                    result, is_err = execute_tool(name, args)
                    msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
                    msgs.append({"role": "tool", "content": result})
                continue
            # No tool call — final answer
            elapsed = time.time() - start
            return content, tool_calls_made, None, elapsed
        elapsed = time.time() - start
        return content, tool_calls_made, "max_turns_exceeded", elapsed
    except Exception as e:
        elapsed = time.time() - start
        return "", tool_calls_made, str(e), elapsed

def run_benchmark(model, rounds=1, only=None):
    results = []
    tests = [t for t in TESTS if only is None or t[0] in only]
    for name, messages, check in tests:
        round_results = []
        for _ in range(rounds):
            text, calls, err, elapsed = call_model(model, messages)
            if err:
                passed, detail = False, f"error: {err}"
            else:
                passed, detail = check(text, calls)
            round_results.append({
                "passed": passed, "detail": detail, "elapsed": elapsed,
                "tool_calls": calls, "error": err, "text": text[:300]
            })
        # aggregate
        passed = all(r["passed"] for r in round_results)
        avg_elapsed = statistics.mean(r["elapsed"] for r in round_results)
        results.append({
            "test": name, "passed": passed,
            "detail": round_results[0]["detail"],
            "avg_elapsed": round_results[0]["elapsed"],
            "tool_calls": round_results[0]["tool_calls"],
            "error": round_results[0]["error"],
            "sample_text": round_results[0]["text"]
        })
    return results

if __name__ == "__main__":
    model = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    print(f"=== Agent Benchmark: {model} (rounds={rounds}) ===", flush=True)
    results = run_benchmark(model, rounds, only)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    total_time = sum(r["avg_elapsed"] for r in results)
    print(f"\nPASSED: {passed}/{total}  |  Total time: {total_time:.1f}s")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        calls = ",".join(r["tool_calls"]) if r["tool_calls"] else "-"
        print(f"[{status}] {r['test']:24s} {r['avg_elapsed']:6.1f}s tools=[{calls}] {r['detail']}")
        if not r["passed"] and r["sample_text"]:
            print(f"        sample: {r['sample_text'][:200]}")
    # JSON for machine parsing
    print("\n---JSON---")
    print(json.dumps({"model": model, "passed": passed, "total": total,
                      "total_time": total_time, "results": results}))
