"""A323: descriptive arithmetic on sealed predictions; no fitting or model calls.

This is a retrospective audit, not a test-set model-selection procedure. All
comparisons below are fixed by the previously observed cases, not searched here.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ARCHIVES = {
    "rotor305": Path("/data/yyf-zyq/PHM_20261010/20261023T052522Z_rotor305"),
    "xjtu278": Path("/data/yyf-zyq/PHM_20261010/20261022T081740Z_feedback278"),
    "ldm321": Path("/data/yyf-zyq/PHM_20261010/20261023T231827Z_gap321_recovered"),
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def cm_metrics(y, pred):
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    diag = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return {
        "n": len(y), "correct": int(diag.sum()),
        "accuracy": float(np.mean(y == pred)),
        "macro_f1": float(np.divide(2 * diag, denom, out=np.zeros(5), where=denom > 0).mean()),
        "confusion": cm.tolist(), "prediction_counts": cm.sum(axis=0).tolist(),
    }


def paired(y, before, after, groups=None):
    out = {
        "before": cm_metrics(y, before), "after": cm_metrics(y, after),
        "changed_labels": int((before != after).sum()),
        "fixed": int(((before != y) & (after == y)).sum()),
        "broken": int(((before == y) & (after != y)).sum()),
        "net_correct": int((after == y).sum() - (before == y).sum()),
    }
    if groups is not None:
        rows = []
        for group in np.unique(groups):
            m = groups == group
            assert len(np.unique(y[m])) == 1
            true = int(y[m][0])
            rows.append({
                "group": str(group), "n": int(m.sum()), "true": true,
                "before_correct": int((before[m] == true).sum()),
                "after_correct": int((after[m] == true).sum()),
                "before_majority_correct": bool(np.bincount(before[m], minlength=5).argmax() == true),
                "after_majority_correct": bool(np.bincount(after[m], minlength=5).argmax() == true),
            })
        out["records"] = rows
        out["record_majority_before"] = sum(x["before_majority_correct"] for x in rows)
        out["record_majority_after"] = sum(x["after_majority_correct"] for x in rows)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if run == Path("/tmp") or Path("/tmp") in run.parents:
        raise ValueError("Use a project run directory, not /tmp")
    for name in ["analysis", "audit"]:
        (run / name).mkdir(parents=True, exist_ok=True)
    bindings_path = run / "audit/SOURCE_BINDINGS323.json"
    bindings = json.loads(bindings_path.read_text()) if bindings_path.exists() else []
    bound = {x["source"] for x in bindings}
    manifests = {}
    for key, root in ARCHIVES.items():
        name = "SOURCE_MANIFEST_RECOVERY3.json" if key == "ldm321" else "SOURCE_MANIFEST.json"
        manifests[key] = {x["path"]: x for x in json.loads((root / name).read_text())["files"]}

    def bind(key, rel, copy=True):
        source = ARCHIVES[key] / rel
        if str(source) not in bound:
            digest = sha(source)
            expected = manifests[key][rel]
            assert digest == expected["sha256"] and source.stat().st_size == expected["bytes"], source
            target = run / "source" / key / rel
            if copy:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            bindings.append({"source": str(source), "copy": str(target.relative_to(run)) if copy else None,
                             "bytes": source.stat().st_size, "sha256": digest,
                             "archive_manifest_verified": True})
            bound.add(str(source))
        return source

    def read(key, rel):
        return json.loads(bind(key, rel).read_text())

    def predictions(key, rel, ids):
        rows = read(key, rel)["predictions"]
        mapping = {x["sample_id"]: int(x["predicted_label"]) for x in rows}
        assert len(mapping) == len(rows) == len(ids) and set(mapping) == set(ids)
        return np.asarray([mapping[i] for i in ids], dtype=int)

    a = "a/workspaces/s305_block34_ordinary_initial/"
    ar = "a_review/workspaces/s305_block34_ordinary_review/"
    for prefix, names in [(a, ["report.md", "solution.py", "train_final.py", "modeling.py"]),
                          (ar, ["report.md", "solution.py", "train_final.py", "modeling.py", "single_model_solution.py"])]:
        for name in names:
            bind("rotor305", prefix + name)
    decision_events = {}
    for stage, job in [("a", "s305_block34_ordinary_initial"), ("a_review", "s305_block34_ordinary_review")]:
        bind("rotor305", f"{stage}/plans/{job}.txt")
        events = bind("rotor305", f"{stage}/primary/{job}/{job}/events.jsonl")
        decision_events[stage] = []
        for lineno, line in enumerate(events.read_text().splitlines(), 1):
            event = json.loads(line)
            item = event.get("item", {})
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                decision_events[stage].append({"line": lineno, "text": item["text"]})
    dump(run / "analysis/PUBLIC_DECISION_EVENTS323.json", decision_events)

    old = np.load(bind("rotor305", ar + ".runtime/review_et_oof.npz"), allow_pickle=False)
    new = np.load(bind("rotor305", ar + ".runtime/final_oof_probabilities.npz"), allow_pickle=False)
    bank = np.load(bind("rotor305", ar + ".runtime/selection/probabilities.npz"), allow_pickle=False)
    for field in ["y", "loads", "groups"]:
        assert np.array_equal(old[field], new[field])
    names = bank["names"].tolist()
    assert np.array_equal(old["y"], bank["y"]) and np.array_equal(old["loads"], bank["loads"])
    mixture = .9 * bank["oof"][:, names.index("et_spectral_leaf2")] + .1 * bank["oof"][:, names.index("logreg_raw")]
    assert np.allclose(mixture, new["probability"], atol=1e-12, rtol=0)
    assert np.array_equal(old["pred"], bank["oof"][:, names.index("et_all_mf005")].argmax(1))
    rotor = {"oof_probability_recipe_max_abs_difference": float(abs(mixture - new["probability"]).max()),
             "oof": paired(old["y"], old["pred"], new["prediction"]), "oof_by_load": []}
    for load in np.unique(old["loads"]):
        mask = old["loads"] == load
        rotor["oof_by_load"].append({"load_Nm": float(load), **paired(old["y"][mask], old["pred"][mask], new["prediction"][mask])})
    mask = old["loads"] > min(old["loads"])
    rotor["oof_except_lowest_load"] = paired(old["y"][mask], old["pred"][mask], new["prediction"][mask])

    for split in ["val", "test"]:
        obs = np.load(bind("rotor305", f"common/block34/restored/ROTORBROKENBAR/data/{split}.npz", copy=False), allow_pickle=False)
        ids = obs["sample_ids"].astype(str).tolist()
        loads = obs["operating_load_Nm"]
        if split == "test":
            truth = {x["sample_id"]: x for x in read("rotor305", "common/block34/restored/ROTORBROKENBAR/hidden/labels.json")}
            y = np.asarray([truth[i]["label"] for i in ids], dtype=int)
            groups = np.asarray([truth[i]["group"] for i in ids])
        else:
            y, groups = obs["labels"], obs["groups"]
        p = predictions("rotor305", f"registered_replays/{split}/a/output/complete.json", ids)
        q = predictions("rotor305", f"registered_replays/{split}/a_review/output/complete.json", ids)
        spec = predictions("rotor305", f"registered_replays/{split}/a_review/output/component_spectral.json", ids)
        lr = predictions("rotor305", f"registered_replays/{split}/a_review/output/component_logistic.json", ids)
        discarded = predictions("rotor305", f"registered_replays/{split}/a/output/discarded_spectral_recipe.json", ids)
        assert np.array_equal(spec, discarded)
        rotor[split] = paired(y, p, q, groups)
        rotor[split]["by_load"] = [{"load_Nm": float(load), **paired(y[loads == load], p[loads == load], q[loads == load], groups[loads == load])} for load in np.unique(loads)]
        rotor[split]["spectral"] = cm_metrics(y, spec)
        rotor[split]["logistic"] = cm_metrics(y, lr)
        rotor[split]["same_prediction_as_spectral"] = int((q == spec).sum())
        rotor[split]["discarded_spectral_recipe_predictions_exactly_match_component"] = True
        broken = (p == y) & (q != y)
        rotor[split]["broken_also_equal_spectral"] = int((broken & (q == spec)).sum())
        rotor[split]["broken_predicted_as_two_bars"] = int((broken & (q == 2)).sum())
        if split == "val":
            assert np.array_equal(bank["vy"], y)
            dev_mixture = .9 * bank["dev"][:, names.index("et_spectral_leaf2")] + .1 * bank["dev"][:, names.index("logreg_raw")]
            assert np.array_equal(q, dev_mixture.argmax(1))
        dump(run / f"analysis/rotor_{split}_aligned323.json", [
            {"sample_id": i, "group": str(g), "load_Nm": float(l), "label": int(t), "initial": int(v), "review": int(w), "spectral": int(s), "logistic": int(r)}
            for i, g, l, t, v, w, s, r in zip(ids, groups, loads, y, p, q, spec, lr)])
    read("rotor305", "common/block34/restored/ROTORBROKENBAR/data/dataset.json")
    reported = read("rotor305", "NUMERICAL_RESULTS305.json")
    for stage, role in [("initial", "before"), ("review", "after")]:
        cohort = "a" if stage == "initial" else "a_review"
        row = next(x for x in reported["rows"] if x["cohort"] == cohort and x["stage"] == stage)
        for metric in ["accuracy", "macro_f1"]:
            assert abs(rotor["test"][role][metric] - row["final_" + metric]) < 1e-12

    # XJTU: unchanged sealed component / full-program outputs, aligned by query.
    xjtu_data = read("xjtu278", "NUMERICAL_RESULTS278.json")
    variants = {x["variant"]["id"]: x["results"] for x in xjtu_data["variants"]}
    xjtu = {"variants": {}, "b_vs_knn_by_bearing": [], "transform_causality_is_not_identified": True}
    for name in ["bank_knn5", "agent_b", "b_tree", "b_mlp", "b_prior_90_10"]:
        xjtu["variants"][name] = {split: variants[name][split]["metrics"] for split in ["development", "final"]}
        for split, metrics in xjtu["variants"][name].items():
            by_unit = {}
            for row in metrics["per_query"]:
                by_unit.setdefault(row["unit_id"], []).append(abs(row["predicted_rul"] - row["actual_rul"]))
            assert abs(np.mean([np.mean(v) for v in by_unit.values()]) - metrics["mean_bearing_mae_minutes"]) < 1e-8
    initial = {x["unit_id"]: x for x in xjtu["variants"]["bank_knn5"]["final"]["per_bearing"]}
    for row in xjtu["variants"]["agent_b"]["final"]["per_bearing"]:
        xjtu["b_vs_knn_by_bearing"].append({"unit_id": row["unit_id"], "knn_mae": initial[row["unit_id"]]["mae_minutes"], "b_mae": row["mae_minutes"], "mae_difference": row["mae_minutes"] - initial[row["unit_id"]]["mae_minutes"]})
    xjtu["saved_component_blend_error"] = {}
    for split in ["development", "final"]:
        aligned = {name: {r["sample_id"]: r for r in xjtu["variants"][name][split]["per_query"]}
                   for name in ["agent_b", "b_tree", "b_mlp"]}
        diffs = [abs(row["predicted_rul"] - .5 * (aligned["b_tree"][key]["predicted_rul"] + aligned["b_mlp"][key]["predicted_rul"]))
                 for key, row in aligned["agent_b"].items()]
        xjtu["saved_component_blend_error"][split] = max(diffs)
        # The submitted neural path uses float32; component replay can differ in rounding.
        assert max(diffs) < 1e-3
    for name in ["report.md", "train_custom.py", "solution.py"]:
        bind("xjtu278", "b/workspaces/s278_adaptive_r1/" + name)

    # LDM: evaluate the claimed development rule and coverage using saved rows.
    ldm_validation = read("ldm321", "l1/workspaces/gap321_l1/validation.json")
    ldm_data = read("ldm321", "EXTERNAL_RESULTS321.json")
    ldm = {"selection": ldm_validation["selection"], "development_observations": ldm_validation["validation_observations"], "candidates": {}, "final_variants": {}}
    for name, value in ldm_validation["candidates"].items():
        ldm["candidates"][name] = {k: v for k, v in value.items() if k in ["validation_metrics", "leave_one_bearing_out"]}
    for variant in ldm_data["variants"]:
        if variant["dataset"] == "LDM_RTF":
            value = variant["metrics"]
            ldm["final_variants"][variant["id"]] = value
            q = value["event_predictions"]
            by_unit = {}
            for row in q:
                by_unit.setdefault(row["unit_id"], []).append(abs(row["predicted_rul"] - row["actual_rul"]))
            assert abs(np.mean([np.mean(x) for x in by_unit.values()]) - value["mean_bearing_mae"]) < 1e-8
    val_y = np.asarray([x["actual_rul"] for x in ldm["development_observations"]])
    final_y = np.asarray([x["actual_rul"] for x in ldm["final_variants"]["agent_l1"]["event_predictions"]])
    ldm["rul_support"] = {"dev_min": float(val_y.min()), "dev_max": float(val_y.max()), "final_min": float(final_y.min()), "final_max": float(final_y.max()), "final_queries_above_dev_max": int((final_y > val_y.max()).sum()), "final_queries": len(final_y)}
    baseline = {x["unit_id"]: x for x in ldm["final_variants"]["fixed_hgb100"]["per_bearing"]}
    ldm["l1_vs_start_by_bearing"] = [
        {"unit_id": row["unit_id"], "start_mae": baseline[row["unit_id"]]["mae"],
         "temperature_mae": row["mae"], "difference": row["mae"] - baseline[row["unit_id"]]["mae"]}
        for row in ldm["final_variants"]["agent_l1"]["per_bearing"]]
    for stage in ["l1", "l2"]:
        for name in ["report.md", "solution.py"]:
            bind("ldm321", f"{stage}/workspaces/gap321_{stage}/" + name)
    bind("ldm321", "l1/workspaces/gap321_l1/build_validation.py")
    read("ldm321", "SPLIT_AND_TARGET_FREEZE321.json")

    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_exploratory_mechanism_audit",
        "researcher_already_knew_final_outcomes": True,
        "new_agent_sessions": 0, "new_fits": 0, "new_model_forward_calls": 0,
        "new_independent_cases": 0, "causal_feedback_effect_identified": False,
        "rotor305_a": rotor, "xjtu278_b": xjtu, "ldm321": ldm,
    }
    dump(run / "analysis/MECHANISM323.json", result)
    dump(bindings_path, bindings)
    # A standalone descriptive figure. Fold models and sealed final models are
    # explicitly separated; no uncertainty bars treating windows as iid units.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    bars = rotor["oof_by_load"] + rotor["val"]["by_load"] + rotor["test"]["by_load"]
    positions = np.arange(len(bars))
    ax.bar(positions - .19, [x["before"]["accuracy"] * 100 for x in bars], .38, label="Before review", color="#34699a")
    ax.bar(positions + .19, [x["after"]["accuracy"] * 100 for x in bars], .38, label="After review", color="#d47b38")
    ax.set_xticks(positions, [str(x["load_Nm"]) for x in bars])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Window accuracy (%)")
    ax.set_xlabel("Operating load (Nm)")
    ax.axvline(4.5, color="#9aa1a8", linestyle="--", linewidth=1)
    ax.axvline(5.5, color="#9aa1a8", linestyle="--", linewidth=1)
    ax.text(2, 111, "Training-load OOF", ha="center", fontsize=9)
    ax.text(5, 111, "Dev", ha="center", fontsize=9)
    ax.text(6.5, 111, "Sealed final", ha="center", fontsize=9)
    ax.text(0, 48, "+114 correct", ha="center", fontsize=9)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Validation improves; unseen higher loads regress", pad=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    cm = np.asarray(rotor["test"]["after"]["confusion"])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=300)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > 165 else "#243444")
    ax.set_xticks(range(5), ["0", "1", "2", "3", "4"])
    ax.set_yticks(range(5), ["0", "1", "2", "3", "4"])
    ax.set_xlabel("Predicted broken bars")
    ax.set_ylabel("Actual broken bars")
    ax.set_title("After review: 1,136 / 1,468 predicted as class 2", pad=13)
    fig.suptitle("A305a rotor: paired audit of previously sealed predictions", fontsize=14, y=1.01)
    fig.text(.01, -.02, "OOF: each load excluded from its fold fit. Dev/final: frozen full-training checkpoints.\nSame laboratory motor; 20 final records. Retrospective decomposition, not a causal intervention.", fontsize=9)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(run / f"analysis/rotor_mechanism323.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"status": "complete", "verified_sources": len(bindings),
                      "rotor_oof_load_net": [(x["load_Nm"], x["net_correct"]) for x in rotor["oof_by_load"]],
                      "rotor_final": {k: v for k, v in rotor["test"].items() if k not in ["records", "by_load"]},
                      "ldm_support": ldm["rul_support"]}, indent=2))


if __name__ == "__main__":
    main()
