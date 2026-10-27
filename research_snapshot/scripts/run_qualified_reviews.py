"""Execute one frozen, qualified review cohort through the existing CLI runner."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

from run_full_benchmark_campaign import clean_env, launch, now, save, status_path, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads((run / "PROTOCOL187.json").read_text())
    jobs = json.loads((run / "PLAN187.json").read_text())["jobs"]
    controller_status = run / protocol.get("controller_status_name", "CONTROLLER_STATUS238.json")
    assert protocol["model"] == "deepseek-flash"
    assert len(jobs) == protocol["max_new_model_sessions"]
    assert 1 <= len(jobs) <= 8
    assert 1 <= protocol["max_new_primary_rolling24h"] <= 400
    assert json.loads((run / protocol.get("training_gate", "TRAINING_QUALIFICATION238.json")).read_text())["passed"]
    assert json.loads((run / protocol.get("environment_gate", "REVIEW_ENVIRONMENT_GATE238.json")).read_text())["passed"]
    assert json.loads((run / protocol.get("authorization_record", "audit/SUBJECT_SCOPE238.json")).read_text())["user_reply"]
    for gate in protocol.get('additional_gates', []):
        assert json.loads((run / gate).read_text())['passed'], 'Additional qualification failed: '+gate
    assert hashlib.sha256(Path(protocol["cli"]).read_bytes()).hexdigest() == protocol["cli_sha256"]
    assert not controller_status.exists(), "No automatic cohort restart"
    for job in jobs:
        assert not (run / "primary" / job["id"]).exists()
        verify(run, job)
    state = {"controller_pid": os.getpid(), "started_utc": now().isoformat(),
             "planned": len(jobs), "completed": [], "launched": []}
    pending, active, serial = list(jobs), {}, 0
    env = clean_env(run)
    while pending or active:
        for ident, (process, job) in list(active.items()):
            if process.poll() is None:
                continue
            path = status_path(run, job)
            status = json.loads(path.read_text()) if path.exists() else {"status": "runner_error"}
            state["completed"].append({"job": ident, "status": status["status"],
                                       "runner_exit_code": process.returncode})
            del active[ident]
        while pending and len(active) < protocol["local_max_concurrency"]:
            serial += 1
            process, admission = launch(run, project, protocol, pending[0], serial, env)
            state["waiting"] = admission if process is None else None
            if process is None:
                break
            job = pending.pop(0)
            active[job["id"]] = process, job
            state["launched"].append(admission)
        state.update(updated_utc=now().isoformat(), pending=len(pending), active_jobs=list(active))
        save(controller_status, state)
        owner = json.loads((run / "OWNERSHIP.json").read_text())
        owner.update(controller_pid=os.getpid(), lease_expires_utc=(now() + timedelta(hours=2)).isoformat(),
                     writers_stopped=False, status="qualified_reviews_running_or_capacity_wait")
        save(run / "OWNERSHIP.json", owner)
        if pending or active:
            time.sleep(5)
    state.update(status="all_selected_terminal_no_scores", ended_utc=now().isoformat())
    save(controller_status, state)
    owner.update(writers_stopped=True, status="terminal_waiting_choice_audit_then_hidden_evaluation")
    save(run / "OWNERSHIP.json", owner)
    print(json.dumps({"status": state["status"], "completed": state["completed"]}))


if __name__ == "__main__":
    main()
