#!/usr/bin/env python3
"""Prepare the frozen A162 ADMITBench D01 pilot without making model calls."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


RUN_ROOT = Path(
    "/srv/yyf-zyq/PHM_20261010/"
    "20261018T062002431304Z_research_check129/A162"
)
SOURCE_ROOT = Path(
    "/data/yyf-zyq/PHM_20261010/"
    "20261017T235800Z_admit152/source/admit-bench"
)
SOURCE_COMMIT = "8b859de77334872efc3fab1c5a764b43d941f210"
CASE_ID = "D01"
CARTRIDGE_ID = "distillation"
JOB_ID = "admit162_d01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(data: object) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()


def write_json(path: Path, data: object) -> None:
    path.write_bytes(json_bytes(data))


def main() -> None:
    sys.path.insert(0, str(SOURCE_ROOT))

    from admitbench.cartridge import load_cartridge
    from admitbench.prompts import SEALED_MARKERS, render_system, render_user

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = RUN_ROOT / "workspaces" / JOB_ID
    workspace.mkdir(parents=True, exist_ok=False)

    cartridge = load_cartridge(CARTRIDGE_ID)
    case = cartridge.case(CASE_ID)
    log = cartridge.evidence_log_for(case)
    context = cartridge.context_for(case)
    system_prompt = render_system(cartridge, style="narrative")
    user_prompt = render_user(cartridge, case, log, context, style="narrative")
    prompt = (
        "## Official ADMITBench system context\n\n"
        + system_prompt
        + "\n\n## Official ADMITBench episode input\n\n"
        + user_prompt
        + "\n"
    )
    leaked = [marker for marker in SEALED_MARKERS if marker in prompt]
    if leaked:
        raise RuntimeError(f"Sealed labels leaked into the participant prompt: {leaked}")

    prompt_path = RUN_ROOT / "PROMPT162.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    public_case = {
        "case_id": case.id,
        "title": case.title,
        "mode": case.mode,
        "hazard_id": case.hazard_id,
        "decision_time_s": case.decided_at(),
        "evidence_visible_at_decision": [
            {
                "id": entry.id,
                "source": entry.source,
                "received_at": entry.received_at,
                "content": entry.content,
                "quality": entry.quality,
            }
            for entry in log.visible_at(context.decided_at)
        ],
        "sealed_fields_absent": list(SEALED_MARKERS),
    }
    write_json(RUN_ROOT / "PUBLIC_CASE162.json", public_case)

    plan = {
        "jobs": [
            {
                "id": JOB_ID,
                "condition": "agent",
                "repeat": 1,
                "prompt_file": "PROMPT162.txt",
                "workspace": f"workspaces/{JOB_ID}",
                "answer_mode": "text",
                "case_ids": [CASE_ID],
            }
        ]
    }
    write_json(RUN_ROOT / "PLAN162.json", plan)

    config_text = """default_permissions = "iot-offline"
model_catalog_json = "/home/jupyter-yyf-zyq/.codex/models.json"
model = "deepseek-flash"
model_provider = "deepseek"
model_reasoning_effort = "max"
web_search = "disabled"
approval_policy = "never"

[model_providers.deepseek]
name = "DeepSeek"
base_url = "https://api.deepseek.com/"
env_key = "DEEPSEEK_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false

[shell_environment_policy]
exclude = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]

[features]
apps = false
multi_agent = false

[permissions.iot-offline]

[permissions.iot-offline.filesystem]
":root" = "write"
":tmpdir" = "write"

[permissions.iot-offline.filesystem.":workspace_roots"]
"." = "write"

[permissions.iot-offline.network]
enabled = false
"""
    (RUN_ROOT / "cli_config_public162.toml").write_text(config_text, encoding="utf-8")
    write_json(RUN_ROOT / "runtime_mounts162.json", {"mounts": []})

    model_catalog = Path(
        "/data/yyf-zyq/PHM_20261010/20261018T003500Z_a153/model_catalog153.json"
    )
    source_files = [
        SOURCE_ROOT / "admitbench/cartridges/distillation/manifest.yaml",
        SOURCE_ROOT / "admitbench/cartridges/distillation/system_graph.jsonl",
        SOURCE_ROOT / "admitbench/cartridges/distillation/safety_case_graph.jsonl",
        SOURCE_ROOT / "admitbench/cartridges/distillation/procedures_cases.jsonl",
        SOURCE_ROOT / "admitbench/prompts.py",
        SOURCE_ROOT / "admitbench/gates.py",
        SOURCE_ROOT / "admitbench/physics.py",
        SOURCE_ROOT / "admitbench/scoring.py",
    ]
    source_lock = {
        "source_root": str(SOURCE_ROOT),
        "repository": "https://github.com/Refiant-Inc/admit-bench",
        "commit": SOURCE_COMMIT,
        "license": "Apache-2.0",
        "cartridge": CARTRIDGE_ID,
        "cartridge_content_hash": cartridge.content_hash(),
        "files": {str(path.relative_to(SOURCE_ROOT)): sha256(path) for path in source_files},
        "model_catalog": str(model_catalog),
        "model_catalog_sha256": sha256(model_catalog),
    }
    write_json(RUN_ROOT / "SOURCE_LOCK162.json", source_lock)

    qualification_root = RUN_ROOT / "qualification_stub_retry1"
    trace_candidates = sorted(qualification_root.glob("D01*.json"))
    qualification = {
        "attempt1": {
            "status": "preflight_wrong_cartridge",
            "command_effect": "admitbench rejected D01 from cartridge cstr",
            "model_calls": 0,
            "output_dir_empty": True,
        },
        "attempt2": {
            "status": "pass",
            "provider": "stub",
            "model": "oracle",
            "cartridge": CARTRIDGE_ID,
            "case_id": CASE_ID,
            "trace": str(trace_candidates[0]) if trace_candidates else None,
            "trace_sha256": sha256(trace_candidates[0]) if trace_candidates else None,
            "purpose": "official end-to-end evaluator and prompt-rendering qualification only",
            "subject_model_calls": 0,
            "hidden_feedback_to_subject": False,
        },
    }
    write_json(RUN_ROOT / "QUALIFICATION162.json", qualification)

    protocol = {
        "run_id": "20261018T062002431304Z_research_check129/A162",
        "experiment_label": "A162",
        "freeze_utc": datetime.now(timezone.utc).isoformat(),
        "role": "trusted_background_research_coordinator_not_phm_subject",
        "status": "prepared_not_launched",
        "purpose": (
            "One deepseek-flash coding-agent episode on the original ADMITBench "
            "distillation case D01, testing whether a diagnosis-correct intervention "
            "also passes the independent T4 physical-consequence gate."
        ),
        "classification": (
            "single_case_pilot_for_natural_trigger_and_mechanism_screen_"
            "not_scoped_problem_confirmation"
        ),
        "case_selection": {
            "case_id": CASE_ID,
            "cartridge": CARTRIDGE_ID,
            "basis": (
                "Published ADMITBench gate results show multiple prior models, including "
                "deepseek-v3.2, diagnosed D01 correctly but failed AAS-T4-ENVELOPE. "
                "Those results only nominate the case; they are not admitted as this "
                "run's denominator or as evidence about deepseek-flash."
            ),
            "natural_task": "choose and parameterize one admissible immediate plant intervention",
            "task_adaptation": (
                "The official narrative system and user prompts are rendered together "
                "into one CLI prompt; one decision point is preserved."
            ),
        },
        "visible_boundary": {
            "participant_sees": [
                "official narrative system prompt",
                "official D01 episode input rendered for the 60-second decision point",
                "action grammar, SOP, trusted evidence feed, and candidate hypotheses",
            ],
            "participant_does_not_see": [
                "oracle and its delta_pct",
                "acceptable_actions",
                "load-bearing evidence labels",
                "diagnosis acceptance list",
                "time-to-hazard",
                "ablations",
                "previous model outputs or scores",
                "gate feedback",
                "source or project history outside the prompt",
            ],
            "prompt_sanitization_assertion": (
                "admitbench.prompts.SEALED_MARKERS are asserted absent before freeze"
            ),
            "workspace_files": "empty; shell, unified exec, web search, and apps disabled",
        },
        "evaluator": {
            "source_commit": SOURCE_COMMIT,
            "source_license": "Apache-2.0",
            "official_modules": [
                "admitbench.prompts",
                "admitbench.parser",
                "admitbench.gates",
                "admitbench.physics",
                "admitbench.scoring",
            ],
            "wrapper": "/home/jupyter-yyf-zyq/Bo_2/PHM_20261010/scripts/score_admit162.py",
            "independence": (
                "The participant process must be terminal and every child process/reader "
                "gone before the wrapper reads response.txt. No score or gate result is "
                "returned to the participant."
            ),
        },
        "participant": {
            "runner": "/home/jupyter-yyf-zyq/Bo_2/PHM_20261010/scripts/replay_diagnostic_cli.py",
            "framework": "codex CLI 0.163.0",
            "cli": (
                "/data/yyf-zyq/PHM_20261010/20261015T215024Z_independent95/"
                "runtime/codex0163/package_files/package/vendor/x86_64-unknown-linux-musl/bin/codex"
            ),
            "cli_sha256": "399ff69c431c066e40de318fba45c2b07f45a76b19939d247ee4828fc70c4215",
            "model_requested": "deepseek-flash",
            "workers": 1,
            "subject_wall_seconds": 1200,
            "cpu_affinity": "0-1",
            "numeric_threads": 2,
            "network_tools": False,
            "credential_source": "/home/jupyter-yyf-zyq/.bashrc",
            "credential_env_name": "DEEPSEEK_API_KEY",
            "credential_values_recorded": False,
            "estimated_run_bytes_upper_bound": 8589934592,
        },
        "quota": {
            "preparation_new_primary_sessions": 0,
            "preparation_model_calls": 0,
            "planned_primary_sessions": 1,
            "rolling_24h_estimate_at_freeze": 119,
            "history_note": "119 is the 04:40 UTC estimate; recount at launch from timestamped registry and actual PIDs",
            "max_concurrent_subject_sessions": 8,
            "rolling_24h_limit": 120,
        },
        "qualification": {
            "source_compiles": True,
            "official_stub_oracle": "D01 admissible, aggregate 1.0",
            "public_prompt_size_bytes": len(prompt.encode()),
            "public_prompt_sha256": sha256(prompt_path),
            "repaired_preflight_errors": 1,
            "repaired_error": "D01 was first requested from CSTR instead of distillation; zero model calls",
        },
        "minimal_discriminating_control": {
            "purpose": (
                "Verify outside the participant that the original T4 gate distinguishes "
                "the D01 oracle magnitude from a same-action but deliberately undersized intervention."
            ),
            "oracle": {"action": "reduce_reboiler_duty", "params": {"delta_pct": 50}},
            "weak": {"action": "reduce_reboiler_duty", "params": {"delta_pct": 10}},
            "rule": (
                "Run exactly once before the primary. Pass if the oracle is admissible and "
                "the 10% action fails at T4; report either outcome without selecting another magnitude."
            ),
        },
        "primary_metrics": [
            "terminal status, exit code, elapsed seconds, response validity",
            "action-record parse success",
            "diagnosis correct from the official T2 gate",
            "hard-gate pass/fail, first failing gate, AAS code",
            "official aggregate and T1-T5 tiers",
            "T4 null/action/oracle trajectory margins and crossing time",
            "requested and observed model identifiers",
        ],
        "decision_rules": {
            "diagnosis_correct_and_admissible": (
                "Evidence against a residual agent-specific action-consequence failure "
                "for this case and model; do not tune a rerun."
            ),
            "diagnosis_correct_and_t4_inadmissible": (
                "Supports a natural diagnosis-to-adequate-action gap in this single case; "
                "retain as candidate only, inspect magnitude/timing and ordinary tool cues, "
                "then test the same mechanism on a preregistered different case."
            ),
            "diagnosis_wrong": (
                "The failure is diagnostic or evidence interpretation, not the target "
                "action-consequence mechanism."
            ),
            "t1_t2_t3_failure": (
                "Record the actual gate; do not relabel it as a T4 magnitude/timing failure."
            ),
            "timeout_or_execution_error": (
                "Keep in the primary denominator, distinguish infrastructure/tool failure "
                "from PHM judgment, and do not count it as a PHM mechanism result."
            ),
            "single_case_scope": (
                "No case, however clear, can enter scoped confirmation without a different "
                "pre-declared case or an appropriate unseen-condition check."
            ),
        },
        "counterhypotheses": [
            "the task is easy and the model will pass",
            "a single prior-failure-nominated case is selection-biased",
            "the deterministic simulator is ground truth by assumption and may not transfer to a real plant",
            "a failed action may reflect prompt/schema misinterpretation rather than PHM planning",
            "codex CLI overhead or the 1200-second budget may truncate an otherwise better decision",
        ],
        "non_claims": [
            "no scoped problem confirmation",
            "no cross-case generality",
            "no field-loss estimate",
            "no claim that published prior-model failures reproduce for deepseek-flash",
        ],
        "stop_conditions": [
            "docs/BACKGROUND_STOP exists",
            "rolling 24-hour primary count would exceed 120",
            "any live PHM primary or scorer exists before launch",
            "prompt sanitation check fails",
            "credentials or hidden fields appear in the participant workspace",
        ],
    }
    protocol_path = RUN_ROOT / "PROTOCOL162.json"
    write_json(protocol_path, protocol)

    blocker_plan = f"""# A162 pre-launch plan

Status: prepared, zero participant model calls.

Source: ADMITBench commit `{SOURCE_COMMIT}`, Apache-2.0.
Case: distillation `D01`, one official single-decision episode.

The published run is used only to nominate a diagnosis-correct/consequence-failing
case. It is not admitted as evidence about `deepseek-flash`.

At launch:

1. Confirm `docs/BACKGROUND_STOP` is absent.
2. Recount the rolling 24-hour primary registry and actual PHM PIDs.
3. Confirm the weak-magnitude control fails T4 and the oracle passes.
4. Run exactly one `deepseek-flash` session for 1200 seconds.
5. Do not read active output or give gate feedback.
6. After all writers and readers stop, score once with the official gates.
7. Archive with manifest SHA verification before interpreting the result.

One case can trigger a mechanism screen; it cannot confirm the research problem.
"""
    (RUN_ROOT / "A162_PLAN.md").write_text(blocker_plan, encoding="utf-8")

    inventory = {
        "files": {
            str(path.relative_to(RUN_ROOT)): sha256(path)
            for path in sorted(RUN_ROOT.rglob("*"))
            if path.is_file()
        }
    }
    write_json(RUN_ROOT / "PREPARE162_MANIFEST.json", inventory)
    print(json.dumps({"status": "prepared", "protocol_sha256": sha256(protocol_path)}, indent=2))


if __name__ == "__main__":
    main()
