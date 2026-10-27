"""Prepare a full-information robot triage audit without exposing test labels."""
import argparse
import csv
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

# Preparation imports immutable archived source and shared dependencies.
sys.dont_write_bytecode = True


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PreBlocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.current = ""
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.active, self.current = True, ""

    def handle_endtag(self, tag):
        if tag == "pre" and self.active:
            self.blocks.append(self.current)
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.current += data


READER = '''"""Read immutable observations and authentic cached reference outputs."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/opt/phm_robot")
sys.path[:0] = ["/opt/phm_sympy", "/opt/phm_fdt"]
def observations(case_id):
    with np.load(ROOT/"observations.npz", allow_pickle=False) as z:
        return z[case_id].copy()
def inventory():
    return list(csv.DictReader((ROOT/"inventory.csv").open()))
def tool_outputs():
    return list(csv.DictReader((ROOT/"reference_outputs.csv").open()))
def replay_dbv(case_ids=None, from_raw=False):
    sys.path.insert(0, str(ROOT/"dbv_source"))
    from src.verifier import residuals_robot as RR, conformal as CF
    from src.robot import run_robot as RUN
    scope=json.loads((ROOT/"tool_scope.json").read_text())
    bank=CF.ConformalBank(names=RR.RESIDUAL_NAMES)
    bank.disp_mask=np.ones(31,dtype=bool)
    with np.load(ROOT/"dbv_calibration.npz",allow_pickle=False) as z:
        bank.centre=z["centre"].copy(); bank.scale=z["scale"].copy()
        bank.cal_scores=z["cal_scores"].copy(); bank.act_ref=float(z["act_ref"])
    verifier=CF.Verifier(RUN.build_fsm(scope["fault_names"]),RR.RESIDUAL_NAMES,scope["fault_names"])
    wanted=set(case_ids) if case_ids else None
    rows=[r for r in inventory() if r["role"]=="query" and (wanted is None or r["case_id"] in wanted)]
    with np.load(ROOT/"features.npz",allow_pickle=False) as z:
        ids=z["case_ids"].tolist(); positions={k:i for i,k in enumerate(ids)}
        feats=np.stack([z["features"][positions[r["case_id"]]] for r in rows])
    if from_raw:
        cols=json.loads((ROOT/"columns.json").read_text())
        cols49=scope["residual_signal_columns"]
        model=RR.RobotResiduals(cols49,lam=scope["ridge_lambda"])
        RR._RFF.clear()
        with np.load(ROOT/"dbv_fit.npz",allow_pickle=False) as z:
            for kind,axis in RR.SPECS:
                stem=kind+"_"+str(axis)
                model.coef[(kind,axis)]=z[stem+"_coef"].copy()
                RR._RFF[(kind,axis)]={n:z[stem+"_"+n].copy() for n in ["W","b","mu","sd"]}
        use=[cols.index(c) for c in cols49]
        feats=np.stack([model.window_features(observations(r["case_id"])[:,use].astype(np.float32)) for r in rows])
    triggers=bank.residual_triggers(feats,0.01)
    out=[]
    signatures=np.column_stack([np.zeros(31,dtype=int),verifier.fsm])
    for row,trig in zip(rows,triggers):
        candidates,tag=verifier.candidates(trig)
        out.append(dict(case_id=row["case_id"],trigger_mask="".join(str(int(x)) for x in trig),
            residual_alarm=bool(trig.any()),candidates=candidates,rule=tag,
            minimum_signature_distance=int(np.abs(signatures-trig[:,None]).sum(0).min())))
    return out
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--case-id",nargs="*");p.add_argument("--replay-dbv",action="store_true");p.add_argument("--from-raw",action="store_true")
    a=p.parse_args()
    if a.replay_dbv: print(json.dumps(replay_dbv(a.case_id,a.from_raw)))
    else:
        rows=tool_outputs()
        if a.case_id: rows=[r for r in rows if r["case_id"] in a.case_id]
        print(json.dumps(rows))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = read(run / "protocol_initial60.json")
    prior = Path(protocol["prior_archive"])
    old = read(prior / "protocol_initial59.json")
    deps = read(prior / "model_dependency59.json")
    sys.path[:0] = [deps["directory"], deps["pyarrow"]["directory"], str(prior / "dbv_source"),
                   read(prior / "sympy_dependency59.json")["directory"], read(prior / "fdt_dependency59.json")["directory"]]
    import numpy as np
    import pyarrow.parquet as pq
    from src.robot import run_robot as RRUN
    from src.verifier import residuals_robot as RR, conformal as CF

    source = run / "sources"
    npm, release = read(source / "npm_latest.json"), read(source / "github_latest.json")
    shared = read(run / "retention_policy.json")["shared_dependencies"]
    cli = Path(shared["path"]) / "node_modules/.bin/codex"
    version = subprocess.run([str(cli), "--version"], check=True, capture_output=True, text=True).stdout.strip()
    assert version == "codex-cli " + npm["version"]
    assert release["tag_name"] == "rust-v" + npm["version"]
    blocks = PreBlocks()
    blocks.feed((source / "deepseek_codex.html").read_text())
    catalogs = []
    for block in blocks.blocks:
        try:
            value = json.loads(block)
            if isinstance(value, dict) and "models" in value:
                catalogs.append(value)
        except ValueError:
            pass
    assert len(catalogs) == 1 and any(x.get("slug") == "deepseek-flash" for x in catalogs[0]["models"])
    save(run / "deepseek_cli_catalog_official.json", catalogs[0])
    fresh_commit = read(source / "dbv_commit.json")["sha"]
    assert fresh_commit == "2dc8d2048095cb158548cfa331890e5956cacb5a"
    save(run / "baseline_qualification60.json", dict(cli_version=version, latest_release_url=release["html_url"],
        release_published_utc=release["published_at"], requested_model="deepseek-flash", provider_url="https://api.deepseek.com/",
        wire_api="responses", official_catalog_sha256=sha(run / "deepseek_cli_catalog_official.json"),
        DBV_commit=fresh_commit, DBV_manuscript_status="public code and author-stated submitted manuscript; not a verified accepted paper",
        service_model_id="native events may expose only request alias; separately record any actual API health response model",
        retrieved_utc=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(run / "runner_frozen.py")))

    data = run / "task_data"
    data.mkdir()
    machine = read(prior / "split_and_shape59.json")["columns"]
    columns = ["time", "action", "active"] + machine
    save(data / "columns.json", columns)
    original = read(prior / "dbv_numeric_execution_attempt2/data/voraus/recordings_meta.json")
    metadata, cols49 = original["meta"], original["columns"]
    normal = [m["sample"] for m in metadata if not m["anomaly"]]
    normal = list(np.random.default_rng(0).permutation(normal))
    fit, cal = set(normal[:300]), set(normal[300:800])
    mvt_train = set(read(prior / "split_and_shape59.json")["training_ids"])
    joint = list(csv.DictReader((prior / "shared_predictive_case_census59.csv").open()))
    assert len(joint) == 870
    queries = {int(r["sample"]) for r in joint}
    known = fit | cal | mvt_train
    assert not (queries & known)
    chosen = sorted(queries | known)
    permuted = list(np.random.default_rng(20261013060).permutation(chosen))
    mapping = {int(sid): "C%04d" % i for i, sid in enumerate(permuted)}
    hidden = run / "hidden"
    hidden.mkdir()
    save(hidden / "source_id_map60.json", mapping)
    gold = [dict(r, case_id=mapping[int(r["sample"])]) for r in joint]
    table(hidden / "query_gold60.csv", gold)
    df = pq.read_table(old["data"]["path"], columns=["sample"] + columns).to_pandas()
    selected = df[df["sample"].isin(chosen)]
    inventory = []
    with zipfile.ZipFile(data / "observations.npz", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for sid, group in selected.groupby("sample", sort=False):
            group = group.sort_values("time", kind="stable")
            values = group[columns].to_numpy(dtype=np.float64)
            assert np.isfinite(values).all()
            buffer = io.BytesIO()
            np.save(buffer, values, allow_pickle=False)
            cid = mapping[int(sid)]
            z.writestr(cid + ".npy", buffer.getvalue())
            inventory.append(dict(case_id=cid, role="query" if sid in queries else "known_normal",
                samples=len(values), mvt_fit=int(sid in mvt_train), dbv_fit=int(sid in fit), dbv_cal=int(sid in cal)))
    inventory.sort(key=lambda r: r["case_id"])
    table(data / "inventory.csv", inventory)
    save(run / "input_integrity60.json", dict(original_data_path=old["data"]["path"], original_data_sha256=sha(Path(old["data"]["path"])),
        observations_sha256=sha(data / "observations.npz"), float64_source_values_preserved=True, columns=columns,
        query_count=len(queries), known_normal_count=len(known), records=len(inventory), samples=sum(r["samples"] for r in inventory),
        removed_fields=["sample (replaced by shuffled opaque ID)", "anomaly", "category", "setting"],
        retained_control_and_motion_phase=["time", "action", "active", "all 130 original machine channels"],
        query_training_overlap=0, independent_robots=1, original_order_is_not_a_chronological_age_clock=True))
    del df, selected

    with np.load(prior / "dbv_cold_features59.npz", allow_pickle=False) as z:
        feats = z["features"].copy()
        ids = z["sample_ids"].tolist()
    positions = {int(k): i for i, k in enumerate(ids)}
    bank = CF.ConformalBank(names=RR.RESIDUAL_NAMES)
    bank.disp_mask = np.ones(31, dtype=bool)
    bank.fit_scale(feats[[positions[x] for x in normal[:300]]]).calibrate(feats[[positions[x] for x in normal[300:800]]])
    np.savez_compressed(data / "dbv_calibration.npz", centre=bank.centre, scale=bank.scale, cal_scores=bank.cal_scores, act_ref=bank.act_ref)
    feature_ids = [r["case_id"] for r in inventory]
    reverse = {v: k for k, v in mapping.items()}
    np.savez_compressed(data / "features.npz", case_ids=np.asarray(feature_ids), features=feats[[positions[reverse[c]] for c in feature_ids]])
    shutil.copy2(prior / "dbv_actual_fit59_1.npz", data / "dbv_fit.npz")
    shutil.copytree(prior / "dbv_source/src", data / "dbv_source/src")
    shutil.copy2(prior / "dbv_source/README.md", data / "dbv_source/README.md")
    shutil.copytree(Path(old["data"]["original_code"]), data / "mvt_source", ignore=shutil.ignore_patterns(".git", "__pycache__"))
    shutil.copy2(prior / "model_final.pth", data / "mvt_final.pth")
    shutil.copy2(prior / "standard_scaler59.json", data / "standard_scaler.json")
    paper = Path('/data/yyf-zyq/PHM_20261010/20261012T181807Z_robot58/sources/original_arxiv_plain58.txt')
    shutil.copy2(paper, data / "voraus_original_paper.txt")
    faults = sorted({r["truth"] for r in joint if r["truth"] != "NF"})
    threshold = read(prior / "reference_metrics_final.json")["train99percentile_threshold"]
    scope = dict(sample_rate_Hz=100, task_time="after complete pick-and-place cycle", query_records=870,
        machine_channels=130, residual_signal_columns=cols49, residual_relations=31, fault_names=faults, ridge_lambda=1e-3,
        MVT=dict(training_normals=948, epochs=70, seed=177, score="negative log likelihood", anomaly_threshold=threshold,
                 threshold_origin="99th percentile of its own normal training scores, not an independent calibrated fault probability",
                 source_commit=old["data"]["code_commit"], model_sha256=sha(data / "mvt_final.pth")),
        DBV=dict(fit_normals=300, calibration_normals=500, alpha=0.01, commit=fresh_commit,
                 mode="nearest structural signature; source strict alternative available in code",
                 guarantee_scope="nominal false-alarm coverage under calibration assumptions; see actual code for candidate semantics"),
        cache_provenance="Both outputs come from actual fitted models with independently verified cold replay. No synthetic expert result.",
        record_labels="Test intervention/category labels are hidden; source names/label IDs masked. Whole-record anomalies, not per-frame onset or lifetime labels.")
    # Read the actual fit hyperparameter instead of inferring it from defaults.
    scope["ridge_lambda"] = read(prior / "dbv_actual_fits59.json")["fits"][0]["lambda_value"]
    save(data / "tool_scope.json", scope)
    outputs = []
    for r in joint:
        outputs.append(dict(case_id=mapping[int(r["sample"])], mvt_score=r["score"], mvt_anomaly=r["prediction"],
            dbv_trigger_mask=r["trigger_mask"], dbv_candidates=r["nearest_candidates"], dbv_rule=r["source_rule_tag"],
            dbv_min_distance=r["minimum_signature_distance"]))
    outputs.sort(key=lambda r: r["case_id"])
    table(data / "reference_outputs.csv", outputs)
    (data / "reference_tools.py").write_text(READER)
    (data / "ACCESS.md").write_text('''The immutable /opt/phm_robot contains complete original 100 Hz recordings for every query and every model fit/calibration record, original time/motion action/active values, all 130 machine channels as exact float64, source code, actual fitted parameter arrays, scopes and authentic reference outputs. inventory.csv lists roles and fit/cal membership. columns.json names the observation columns. Each observations.npz member is one unpadded full cycle named by case_id. Physical signal units and motion/control roles are described in voraus_original_paper.txt; source code provides complete model definitions. Derived drive/controller quantities are retained, not re-labelled as independent hardware sensors.

The query subset is the whole common held-out cohort of these two fixed models, rather than a sample of tool disagreements. It is from one robot, with an intentional mixture of nominal and operator-staged anomaly conditions; the mixture is not a deployment prevalence sample. Source labels and IDs that encode them are excluded. No per-frame anomaly-onset gold, natural wear age, maintenance-effect oracle or uniquely identifiable cause is promised.

Use sys.path.insert(0,"/opt/phm_robot"); import reference_tools. observations(case_id) reads an original full cycle; tool_outputs() reads authenticated model results; replay_dbv() reconstructs residual thresholds and original signature rules using actual cached features. replay_dbv([id],from_raw=True) also recomputes all residual features from original signals and actual learned parameters. The cached MVT scores are from its actual final 70-epoch weights, not probabilities. Full weights, scaler and original architecture are available if further inference is needed. Numerical libraries, CPU torch and source dependencies are mounted read-only; no GPU or external data is required. Tool outputs are optional evidence; write your own analysis if useful.
''')
    mounts = [{"source": str(data), "destination": "/opt/phm_robot"},
              {"source": deps["directory"], "destination": "/opt/phm_python/base"},
              {"source": deps["pyarrow"]["directory"], "destination": "/opt/phm_python/extra"},
              {"source": str(prior / "FrEIA_source"), "destination": "/opt/phm_python/jax"},
              {"source": read(prior / "sympy_dependency59.json")["directory"], "destination": "/opt/phm_sympy"},
              {"source": read(prior / "fdt_dependency59.json")["directory"], "destination": "/opt/phm_fdt"}]
    save(run / "runtime_mounts60.json", {"mounts": mounts})
    # The inherited runner already supplies these numeric package paths.
    tasks = run / "tasks"
    tasks.mkdir()
    for repeat in (1, 2):
        task = tasks / ("TRIAGE_r%d" % repeat)
        (task / "inputs").mkdir(parents=True)
        shutil.copy2(data / "ACCESS.md", task / "inputs/ACCESS.md")
    prompt = '''你负责一批工业六轴机器人完整抓取循环的 PHM 筛查审查。请基于 /opt/phm_robot 的原始信号、已拟合模型、实际参考工具输出及实现，形成可复现的异常分诊表与证据报告。输入说明见 inputs/ACCESS.md。可以使用完整传感器/控制信息和正常拟合/校准记录，检查工具代码与运行结果；评价标签不提供。所有870条 query 都要保留，不按工具是否成功过滤。预测时点是每个完整循环结束，时间范围保持在这个任务内。

交付 analysis.py，运行 python3 analysis.py 后从空输出目录生成 predictions.csv、qualification.json、report.md。predictions.csv 每条query一行，字段至少包括 case_id, anomaly_decision（normal/abnormal/unknown）, candidate_faults（竖线分隔、可空或多候选）, evidence_status（自定义可解释证据状态）, physical_health_certified（true/false）, reason。请给出基于现有证据的分诊，并在报告中明确每种结论与动作建议的依据，以及两工具的适用范围、候选或告警之间关系。候选标签是工具/任务假设，不要求唯一物理原因；允许未知，解释无法作进一步结论的原因。qualification.json 记录信息范围、工具依据、正常拟合/校准使用以及分诊汇总。保留成功与失败可能性，原输入不可修改。

报告和数值结果应由同一脚本重建，依据应可在输入或计算中复核。你有1800秒总时限、4个数值CPU线程、无GPU。现有工具与输出可立即使用，无需重训大型模型。在时限内完成并提交合法JSON元数据：results 仅包含 case_id=ROBOT_TRIAGE, status, analysis_script, predictions_file, qualification_file, report, summary 的一项。
'''
    (run / "native_prompt60.txt").write_text(prompt)
    schema = {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object",
        "properties": {k: {"type": "string"} for k in ["case_id", "status", "analysis_script", "predictions_file", "qualification_file", "report", "summary"]},
        "required": ["case_id", "status", "analysis_script", "predictions_file", "qualification_file", "report", "summary"], "additionalProperties": False}}},
        "required": ["results"], "additionalProperties": False}
    save(run / "native_schema60.json", schema)
    save(run / "native_plan60.json", {"jobs": [dict(id="TRIAGE_r%d" % repeat, condition="complete_original_observations_and_real_tools", repeat=repeat,
        workspace="tasks/TRIAGE_r%d" % repeat, prompt_file="native_prompt60.txt", schema_file="native_schema60.json", case_ids=["ROBOT_TRIAGE"]) for repeat in (1, 2)]})
    common = ["python3", str(run / "runner_frozen.py"), "--cli", str(cli), "--model", "deepseek-flash",
        "--analysis-tools", "--isolation-config", str(run / "cli_config_catalog.toml"), "--credential-env-file", "/home/jupyter-yyf-zyq/.bashrc",
        "--model-catalog", str(run / "deepseek_cli_catalog_official.json"), "--runtime-mounts-file", str(run / "runtime_mounts60.json"),
        "--snapshot-generated-source", "--cleanup-runtime-cache"]
    save(run / "native_command60.json", {"cwd": str(run), "argv": common + ["--plan", str(run / "native_plan60.json"), "--output", str(run / "native"), "--workers", "2", "--timeout", "1800"]})
    save(run / "prepared60.json", {"status": "finished", "query_count": 870, "normal_training_pool": len(known),
        "source_double_values_preserved": True, "data_bytes": sum(p.stat().st_size for p in data.rglob("*") if p.is_file()),
        "new_LLM_calls": 0, "at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(read(run / "prepared60.json")), flush=True)


if __name__ == "__main__":
    main()
