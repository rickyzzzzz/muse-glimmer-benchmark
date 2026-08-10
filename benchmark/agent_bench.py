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

def run_benchmark(model, rounds=1):
    results = []
    for name, messages, check in TESTS:
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
    print(f"=== Agent Benchmark: {model} (rounds={rounds}) ===", flush=True)
    results = run_benchmark(model, rounds)
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
