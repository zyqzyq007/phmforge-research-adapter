"""Host-cron checks and a bounded research coordinator, separate from subjects."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import threading
from zoneinfo import ZoneInfo

WORKSPACE = Path(__file__).resolve().parents[1]
CONFIG = WORKSPACE / "configs/background_research.json"
STATE = WORKSPACE / "docs/background_execution_state.json"


def coordinator_lock_path(workspace=WORKSPACE):
    # A review may invoke an archiver or primary admission in a child process.
    # Those operations use background_research.lock for their short transactions;
    # holding that same lock for the entire review makes its child wait on itself.
    return Path(workspace) / "logs/background_coordinator.lock"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + str(os.getpid()) + ".new")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def load(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def active(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, TypeError, ValueError):
        return False


def root_owned_runs(config, checked_at=None):
    """Keep a review from taking over a live foreground ownership lease."""
    checked_at = checked_at or datetime.now(timezone.utc)
    protected = []
    for path in sorted(Path(config["run_root"]).glob("*/OWNERSHIP.json")):
        try:
            lease = json.loads(path.read_text())
            if not isinstance(lease, dict):
                raise ValueError("ownership metadata must be an object")
            if lease.get("owner") != "root" or lease.get("takeover_allowed") is True:
                continue
            expires = datetime.fromisoformat(lease["lease_expires_utc"])
            if expires > checked_at:
                protected.append({"path": str(path), "lease_expires_utc": expires.isoformat()})
        except (OSError, ValueError, TypeError, KeyError):
            protected.append({"path": str(path), "reason": "invalid_ownership_metadata"})
    return protected


def snapshot(config):
    rows = []
    for run in sorted(Path(config["run_root"]).iterdir()):
        if not run.is_dir() or run.name.startswith("_"):
            continue
        for base, dirs, files in os.walk(run):
            dirs[:] = [x for x in dirs if x not in {
                "workspace_snapshot", "source_history", "clients-cache", ".git",
                "deps", "node_modules", "__pycache__", "codex_home"}]
            if "status.json" not in files:
                continue
            path = Path(base) / "status.json"
            item = load(path, {})
            if not isinstance(item, dict) or not item.get("job_id") or not item.get("cli_pid"):
                continue
            rows.append({
                "path": str(path), "job_id": item["job_id"],
                "role": "judge" if "_judge" in item["job_id"] else "primary",
                "status": item.get("status"), "started_utc": item.get("started_utc"),
                "ended_utc": item.get("ended_utc"),
                "pid": item["cli_pid"], "pid_alive": active(item["cli_pid"]),
            })
    return rows


def progress(config, state, rows):
    local = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    primary = [x for x in rows if x["role"] == "primary"]
    running = [x for x in rows if x["pid_alive"] and x["status"] == "running"]
    report = [
        "# PHM 后台实验检查", "", "检查时间：" + local, "",
        "此表为机械运行状态，不是准确率、物理失败数或独立设备数。",
        "研究判断见 docs/background_findings.md；当前批次见 docs/discovery_state.json。",
        "", "- 活跃运行目录中的 primary 会话记录：" + str(len(primary)),
        "- 实际存活且状态 running 的会话：" + str(len(running)),
        "- 检查状态：" + state["monitor_status"],
        "- 上次研究复盘：" + str(state.get("last_review_started_utc", "尚未运行")),
        "- 复盘状态：" + str(state.get("last_review_status", "尚未运行")),
        "- STOP 文件：" + config["stop_file"], "",
    ]
    for row in running:
        report.append("- " + row["job_id"] + "；PID " + str(row["pid"]))
    path = WORKSPACE / "docs/background_progress.md"
    temp = path.with_name(path.name + "." + str(os.getpid()) + ".new")
    temp.write_text("\n".join(report) + "\n")
    temp.replace(path)


def secret_from_existing_configuration():
    pattern = re.compile(r"^\s*(?:export\s+)?DEEPSEEK_API_KEY\s*=\s*(.*)$")
    values = [m.group(1) for line in Path("/home/jupyter-yyf-zyq/.bashrc").read_text().splitlines()
              if (m := pattern.match(line))]
    if len(values) != 1:
        raise ValueError("Expected one existing credential assignment")
    parts = shlex.split(values[0], comments=True)
    if len(parts) != 1 or not parts[0] or any(c in parts[0] for c in ["$", chr(96), "\n"]):
        raise ValueError("Expected a literal runtime credential")
    return parts[0]


def review(config, state, probe=False):
    binary = Path(config["cli"])
    if hashlib.sha256(binary.read_bytes()).hexdigest() != config["cli_sha256"]:
        raise ValueError("Frozen CLI binary changed")
    key = secret_from_existing_configuration()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = Path(config["run_root"]) / (stamp + ("_controller_probe129" if probe else "_research_check129"))
    runtime = run / "runtime"
    for name in ["tmp", "mpl", "codex_home"]:
        (runtime / name).mkdir(parents=True, exist_ok=True)
    public_config = (WORKSPACE / config["cli_config"]).read_text()
    (runtime / "codex_home/config.toml").write_text(public_config)
    save(run / "scheduler_config.json", config)
    prompt = ("This is a non-PHM controller connectivity probe. Use the shell once "
              "to calculate sum([2,3,5]). Return only CONTROLLER_PROBE_OK sum=10. "
              "Do not read credentials or change project files."
              if probe else (WORKSPACE / config["review_prompt"]).read_text())
    (run / "prompt.txt").write_text(prompt)
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ["KEY", "TOKEN", "SECRET", "PASSWORD"])}
    env.update(DEEPSEEK_API_KEY=key, CODEX_HOME=str(runtime / "codex_home"),
               TMPDIR=str(runtime / "tmp"), TMP=str(runtime / "tmp"),
               TEMP=str(runtime / "tmp"), MPLCONFIGDIR=str(runtime / "mpl"),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(name, None)
    limit = 90 if probe else config["review_timeout_seconds"]
    command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s", str(limit) + "s",
               str(binary), "exec", "--ephemeral", "--skip-git-repo-check", "--json",
               "--disable", "apps", "--disable", "multi_agent",
               "-m", "deepseek-flash", "-c", 'web_search="disabled"',
               "-c", 'approval_policy="never"', "-C", str(WORKSPACE), "-"]
    record = {"role": "trusted_research_coordinator_not_eval_subject",
              "model_requested": "deepseek-flash", "model_returned": "not_independently_exposed",
              "started_utc": now(), "status": "running", "timeout_seconds": limit,
              "command": command, "credential_values_recorded": False,
              "env_names_only": sorted(env), "config_sha256": hashlib.sha256(public_config.encode()).hexdigest(),
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env, start_new_session=True)
    record["pid"] = process.pid
    save(run / "controller_status.json", record)
    state.update(last_review_started_utc=record["started_utc"],
                 last_review_status="running", controller_pid=process.pid,
                 controller_run=str(run))
    save(STATE, state)

    def drain(stream, path, collect=False):
        written = 0
        with path.open("w") as output:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", "replace").replace(key, "[REDACTED_CREDENTIAL]")
                if written < 32 * 1024 * 1024:
                    output.write(line)
                    output.flush()
                    written += len(line.encode())
                if collect:
                    try:
                        event = json.loads(line)
                        item = event.get("item", {})
                        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                            (run / "response.md").write_text(item.get("text", ""))
                    except (json.JSONDecodeError, TypeError):
                        pass

    threads = [threading.Thread(target=drain, args=(process.stdout, run / "events.jsonl", True)),
               threading.Thread(target=drain, args=(process.stderr, run / "stderr.txt"))]
    for thread in threads:
        thread.start()
    try:
        process.stdin.write(prompt.encode())
        process.stdin.close()
        code = process.wait(timeout=limit + 15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        code = process.wait()
    except BrokenPipeError:
        code = process.wait(timeout=limit + 15)
    finally:
        for thread in threads:
            thread.join()
    record.update(status="finished" if code == 0 else "timeout" if code in [124, 137] else "execution_error",
                  exit_code=code, ended_utc=now())
    save(run / "controller_status.json", record)
    import shutil
    shutil.rmtree(runtime / "codex_home", ignore_errors=True)
    (run / "public_cli_config.toml").write_text(public_config)
    save(run / "retention_policy.json", {
        "excluded_rebuildable_directory": "clients-cache",
        "shared_dependencies": {"path": str(Path(config["run_root"]) / "_shared"),
                                "retention": "existing pinned readonly dependencies"}})
    state.update(last_review_status=record["status"], last_review_ended_utc=record["ended_utc"],
                 controller_pid=None, controller_run=str(run))
    save(STATE, state)
    with (run / "archive_command.log").open("w") as log:
        result = subprocess.run(["/usr/bin/python3", "-B", str(WORKSPACE / "scripts/archive_run_locked.py"),
                                 "--run", str(run), "--workspace", str(WORKSPACE)],
                                stdout=log, stderr=subprocess.STDOUT,
                                env={k: v for k, v in env.items() if k != "DEEPSEEK_API_KEY"})
    if result.returncode:
        state["last_archive_status"] = "failed_source_retained"
    else:
        state.update(last_archive_status="verified_source_cleared",
                     last_controller_archive=str(Path(config["archive_root"]) / run.name))
    save(STATE, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tick", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--from-cron", action="store_true")
    args = parser.parse_args()
    config = load(CONFIG, {})
    lock_path = coordinator_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if args.from_cron:
        with (WORKSPACE / "logs/background_cron_invocations.jsonl").open("a") as log:
            log.write(json.dumps({"utc": now(), "pid": os.getpid(), "parent_pid": os.getppid()}) + "\n")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # A model review must not block the independent five-minute monitor.
            state = load(STATE, {})
            state["monitor_status"] = "review_in_progress_monitor_only"
            rows = snapshot(config)
            save(WORKSPACE / "logs/background_latest_sessions.json", rows)
            save(WORKSPACE / "docs/background_monitor_state.json", {
                "checked_utc": now(), "review_in_progress": True,
                "sessions": len(rows), "controller_pid": state.get("controller_pid")})
            progress(config, state, rows)
            return
        state = load(STATE, {})
        state.update(last_monitor_utc=now(), monitor_status="checked")
        rows = snapshot(config)
        save(WORKSPACE / "logs/background_latest_sessions.json", rows)
        protected = root_owned_runs(config)
        state["root_owned_review_deferrals"] = protected
        if not config.get("enabled", False):
            state["monitor_status"] = "disabled_no_new_review"
        elif Path(config["stop_file"]).exists():
            state["monitor_status"] = "STOP_set_no_new_review"
        elif active(state.get("controller_pid")):
            state["monitor_status"] = "existing_coordinator_alive"
        elif protected:
            state["monitor_status"] = "root_ownership_lease_monitor_only"
        elif args.probe:
            previous_review = {k: state.get(k) for k in [
                "last_review_started_utc", "last_review_ended_utc", "last_review_status",
                "controller_pid", "controller_run"]}
            review(config, state, probe=True)
            state["controller_probe"] = {
                "status": state["last_review_status"],
                "ended_utc": state["last_review_ended_utc"],
                "archive": state.get("last_controller_archive")}
            state.update(previous_review)
        elif args.tick and not config.get("bootstrap_active", False):
            previous = state.get("last_review_started_utc")
            due = not previous or (datetime.now(timezone.utc) -
                                   datetime.fromisoformat(previous)).total_seconds() >= config["review_interval_seconds"]
            if due:
                review(config, state)
        elif config.get("bootstrap_active", False):
            state["monitor_status"] = "bootstrap_lanes_owned_by_current_agents"
        state["last_monitor_utc"] = now()
        save(STATE, state)
        progress(config, state, snapshot(config))


if __name__ == "__main__":
    main()
