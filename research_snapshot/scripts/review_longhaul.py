#!/usr/bin/env python3
"""Exploratory generic review of a frozen original LongHaul conversation.

This is a researcher-added countercheck, not the author's original workflow.
The source parser, Tools and terminal scorer are reused unchanged. Credentials
are supplied by the caller's environment; they are never included in messages.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review(author, transport, endpoint, world, episode, prior_trace, instruction,
           max_steps=6):
    """Continue the complete terminal conversation without correctness feedback."""
    messages = json.loads(json.dumps(prior_trace))
    messages.append({"role": "user", "content": instruction})
    tools = author.Tools(world, episode["machine_id"], episode["alarms"])
    tokens, tool_calls, anomalies = {"prompt": 0, "completion": 0}, 0, 0
    component, mode, status = "", "", "no_diagnosis"
    started = time.time()
    for _ in range(max_steps):
        if transport.soft_stop_reasons():
            status = "review_budget_stop"
            break
        content, usage = transport(endpoint, messages)
        tokens["prompt"] += usage.get("prompt_tokens", 0)
        tokens["completion"] += usage.get("completion_tokens", 0)
        obj = author.parse_json(content)
        if obj is None:
            anomalies += 1
            if anomalies > 1:
                break
            messages.extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Invalid output. Respond with ONE valid JSON object only."},
            ])
            continue
        if "diagnosis" in obj:
            diagnosis = obj["diagnosis"] or {}
            component, mode = diagnosis.get("component", ""), diagnosis.get("failure_mode", "")
            messages.append({"role": "assistant", "content": content})
            status = "ok"
            break
        if "tool" in obj and hasattr(tools, str(obj["tool"])):
            tool_calls += 1
            result = getattr(tools, obj["tool"])(**(obj.get("args") or {}))
            messages.extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Tool result: {json.dumps(result)}"},
            ])
        else:
            anomalies += 1
            messages.extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Unknown tool. Use alarm_lookup, manual_search, maintenance_history, or give a diagnosis."},
            ])
    result = author._result(episode, component, mode, started, tokens, tool_calls, anomalies, status)
    result["trace"] = messages
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    adapter = load_module(protocol["adapter"], "review_transport")
    for item in protocol["inputs"]:
        if adapter.sha(item["path"]) != item["sha256"]:
            raise ValueError("Frozen input changed: " + item["path"])
    out = Path(protocol["out"])
    out.mkdir(parents=True, exist_ok=False)
    author = adapter.load_author(protocol["author_source"])
    world = json.loads(Path(protocol["world"]).read_text())
    episodes = {x["episode_id"]: x for x in
                map(json.loads, Path(protocol["episodes"]).read_text().splitlines())}
    labels = {x["episode_id"]: x for x in
              map(json.loads, Path(protocol["labels"]).read_text().splitlines())}
    transport = adapter.Transport(out, protocol["model"], os.environ["DEEPSEEK_API_KEY"],
                                  started_epoch=time.time(), budgets=protocol["budgets"])
    adapter.save(out / "status.json", {"status": "running", "started_utc": adapter.utc(),
                                       "protocol_sha256": adapter.sha(args.protocol)})

    def worker(item):
        case_id = item["episode_id"]
        transport.begin_case(case_id)
        prior = json.loads(Path(item["prior_result"]).read_text())
        episode = dict(episodes[case_id])
        # Private truth is used only by the original terminal _result scorer.
        episode["ground_truth"] = labels[case_id]["ground_truth"]
        try:
            result = review(author, transport, protocol["model"]["endpoint"], world,
                            episode, prior["trace"], protocol["instruction"],
                            protocol["budgets"]["max_steps_per_episode"])
            transport.write(out / "cases" / case_id / "review_result.json", result)
            state = {"episode_id": case_id, "status": "terminal", "result_status": result["status"]}
        except Exception as exc:
            state = {"episode_id": case_id, "status": "execution_error",
                     "error_type": type(exc).__name__, "error": str(exc)}
        state["usage"] = dict(transport.local.counts)
        transport.write(out / "cases" / case_id / "execution.json", state)
        return state

    with cf.ThreadPoolExecutor(max_workers=protocol["concurrency"]) as pool:
        states = list(pool.map(worker, protocol["cases"]))
    transport.write(out / "status.json", {"status": "terminal", "ended_utc": adapter.utc(),
                                          "protocol_sha256": adapter.sha(args.protocol),
                                          "cases": states, "usage": transport.snapshot()})
    print(json.dumps({"status": "terminal", "cases": len(states), "usage": transport.snapshot()}))


if __name__ == "__main__":
    main()
