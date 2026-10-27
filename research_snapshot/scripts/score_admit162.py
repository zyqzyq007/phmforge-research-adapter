#!/usr/bin/env python3
"""Score one frozen A162 ADMITBench response with the original deterministic gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-id", default="D01")
    parser.add_argument("--cartridge", default="distillation")
    parser.add_argument("--expected-verdict", choices=["admissible", "inadmissible", "not_evaluable"])
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root))
    from admitbench.cartridge import load_cartridge
    from admitbench.gates import run_gates
    from admitbench.hypotheses import candidate_hypotheses
    from admitbench.parser import parse_action_record
    from admitbench.runner import _escalation_outcome
    from admitbench.record import BOLDNESS
    from admitbench.scoring import score_episode

    raw = args.response.read_text(encoding="utf-8")
    record, parse_error = parse_action_record(raw)
    cartridge = load_cartridge(args.cartridge)
    case = cartridge.case(args.case_id)
    log = cartridge.evidence_log_for(case)
    context = cartridge.context_for(case)
    world = cartridge.world()
    gate_report, artifacts = run_gates(
        record,
        case,
        cartridge.rulebook,
        log,
        context,
        world,
        parse_error=parse_error,
        horizon_s=cartridge.horizon_s,
    )
    artifacts["hypotheses"] = [
        hypothesis.to_dict() for hypothesis in candidate_hypotheses(cartridge, log, context)
    ]
    if case.oracle:
        initial = world.initial_state(case.initial_state)
        artifacts["oracle_trajectory"] = world.project(
            initial,
            case.oracle["action"],
            case.oracle.get("params"),
            horizon_s=cartridge.horizon_s,
        ).summary()
    if record is not None:
        rule = cartridge.rulebook.get(record.action)
        if rule is not None and BOLDNESS[rule.action_class] <= BOLDNESS["escalate"]:
            artifacts["escalation_outcome_trajectory"] = _escalation_outcome(
                world,
                world.initial_state(case.initial_state),
                case,
                context.response_time_s,
                cartridge.horizon_s,
            )
    score = score_episode(
        record,
        case,
        cartridge.rulebook,
        gate_report,
        artifacts,
        cartridge.weights,
    )
    t2 = gate_report.gate("T2")
    result = {
        "case_id": case.id,
        "cartridge": cartridge.id,
        "cartridge_content_hash": cartridge.content_hash(),
        "source_commit": "8b859de77334872efc3fab1c5a764b43d941f210",
        "response": str(args.response),
        "response_sha256": sha256(args.response),
        "raw_response": raw,
        "parse_error": parse_error,
        "record": record.to_dict() if record else None,
        "diagnosis_correct": bool(t2 and t2.info.get("diagnosed")),
        "gates": gate_report.to_dict(),
        "artifacts": artifacts,
        "score": score.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": score.verdict,
                "diagnosis_correct": result["diagnosis_correct"],
                "first_failing": score.first_failing,
                "aas_code": score.aas_code,
                "aggregate": score.aggregate,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if args.expected_verdict and score.verdict != args.expected_verdict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
