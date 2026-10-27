"""Prepare one complete pass over version-pinned public benchmark questions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


CLIENT = Path(__file__).with_name('tool_bridge_client.py').read_text()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forge_command(run, workspace, project):
    command = ['bwrap', '--unshare-user', '--unshare-pid', '--unshare-ipc',
               '--unshare-uts', '--unshare-net', '--die-with-parent', '--new-session']
    for path in ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc',
                 str(Path.home() / '.local/lib')]:
        if Path(path).exists():
            command += ['--ro-bind', path, path]
    for source, target in [(run / 'runtime/forge_public', '/opt/forge'),
                           (run / 'runtime/forge_deps', '/opt/forge_deps'),
                           (run / 'runtime/forge_data', '/opt/forge_data'),
                           (project / 'scripts/benchmark_tool_bridge.py', '/worker.py')]:
        command += ['--ro-bind', str(source), target]
    command += ['--bind', str(workspace), '/app', '--proc', '/proc', '--dev', '/dev',
                '--chdir', '/app', '--setenv', 'PYTHONPATH', '/opt/forge_deps:/opt/forge',
                '--setenv', 'PHMFORGE_DATA_DIR', '/opt/forge_data',
                '--setenv', 'PYTHONDONTWRITEBYTECODE', '1',
                '--setenv', 'TMPDIR', '/app/.runtime/tooltmp',
                '--setenv', 'TMP', '/app/.runtime/tooltmp',
                '--setenv', 'TEMP', '/app/.runtime/tooltmp',
                '--setenv', 'MPLCONFIGDIR', '/app/.runtime/mpl',
                '--setenv', 'XDG_CACHE_HOME', '/app/.runtime/cache',
                '--setenv', 'HF_HOME', '/app/.runtime/hf',
                '--setenv', 'HF_HUB_OFFLINE', '1',
                '--setenv', 'TRANSFORMERS_OFFLINE', '1',
                '--setenv', 'OMP_NUM_THREADS', '1',
                '--setenv', 'OPENBLAS_NUM_THREADS', '1',
                'python3', '-B', '/worker.py', 'worker', '--family', 'forge']
    return command


def asset_command(run, workspace, project, couch, job):
    runtime = run / 'runtime'
    command = ['docker', 'run', '--rm', '-i', '--name', 'phm187_tool_' + job,
               '--network', couch['network'], '--user', f'{os.getuid()}:{os.getgid()}',
               '--cpus', '2', '--memory', '4g',
               '-v', str(runtime / 'asset_deps') + ':/deps:ro',
               '-v', str(runtime / 'asset_public') + ':/opt/asset/src:ro',
               '-v', str(project / 'scripts/benchmark_tool_bridge.py') + ':/worker.py:ro',
               '-v', str(workspace) + ':/app', '-w', '/app']
    environment = dict(PYTHONPATH='/deps:/opt/asset/src', PYTHONDONTWRITEBYTECODE='1',
                       TMPDIR='/app/.runtime/tooltmp', TMP='/app/.runtime/tooltmp',
                       TEMP='/app/.runtime/tooltmp', MPLCONFIGDIR='/app/.runtime/mpl',
                       XDG_CACHE_HOME='/app/.runtime/cache', HF_HOME='/app/.runtime/hf',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       COUCHDB_URL='http://' + couch['name'] + ':5984',
                       COUCHDB_USERNAME='admin', COUCHDB_PASSWORD='password',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key, value in environment.items():
        command += ['-e', key + '=' + value]
    command += ['python:3.13-slim', 'python', '-B', '/worker.py', 'worker', '--family', 'asset']
    return command


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run, project = args.run.resolve(), Path(__file__).resolve().parents[1]
    inventory = json.loads((run / 'TASK_INVENTORY187.json').read_text())
    if (run / 'PLAN187.json').exists():
        raise FileExistsError('Refusing to replace existing campaign')
    (run / 'plans').mkdir()
    (run / 'workspaces').mkdir()
    (run / 'primary').mkdir()
    (run / 'logs').mkdir(exist_ok=True)
    (run / 'traces').mkdir()
    (run / 'ipc').mkdir()
    shutil.copy2('/data/yyf-zyq/PHM_20261010/20261018T142647Z_multiscope185/cli_config185.toml',
                 run / 'cli_config187.toml')
    shutil.copy2('/data/yyf-zyq/PHM_20261010/20261018T003500Z_a153/model_catalog153.json',
                 run / 'model_catalog187.json')
    resources = json.loads((run / 'runtime/DOCKER_RESOURCES187.json').read_text())
    couch = {x['scenario']: dict(network=resources['network'], **x)
             for x in resources['couch_containers']}
    routes, jobs, immutable = {}, [], []
    selections = [('forge', row) for row in inventory['phmforge']['rows']]
    selections += [('asset', row) for row in inventory['assetops_main']['rows'] if not row['missing']]
    for index, (family, row) in enumerate(selections, 1):
        # Numeric job identifiers do not trigger old capacity filters that reserve
        # names containing 'health' for connectivity probes.
        job_id = family + '_' + str(index).zfill(4)
        workspace = run / 'workspaces' / job_id
        workspace.mkdir()
        (workspace / 'inputs').mkdir()
        for sub in ['tooltmp', 'mpl', 'cache', 'hf']:
            (workspace / '.runtime' / sub).mkdir(parents=True)
        capability = uuid.uuid4().hex
        (workspace / 'inputs/tool_connection.json').write_text(json.dumps(dict(route=capability)) + '\n')
        (workspace / 'tool_client.py').write_text(CLIENT)
        (workspace / 'inputs/question.txt').write_text(row['question'] + '\n')
        (workspace / 'readonly_inputs.json').write_text(json.dumps(['tool_client.py']) + '\n')
        instructions = '''You are completing the following original industrial asset benchmark task.
Use the original benchmark tools available through the local Python client.

Tool discovery: python3 tool_client.py list_tools '{}'
Optional server prefix: python3 tool_client.py list_tools '{"prefix":"wo."}'
Tool schema: python3 tool_client.py tool_schema '{"name":"TOOL_NAME"}'
Tool invocation: python3 tool_client.py TOOL_NAME '{"argument":"value"}'

The client returns actual original tool results. You may write and execute analysis
code in /app and save data files there. Tool file arguments use the same /app paths.
Provide the final result requested by the original task. Work files and caches
must stay under /app; use .runtime/tooltmp for temporary files.

Original task (unchanged):
'''
        prompt = workspace / 'instruction.txt'
        prompt.write_text(instructions + row['question'] + '\n')
        tool_command = (forge_command(run, workspace, project) if family == 'forge'
                        else asset_command(run, workspace, project, couch[row['task_id']], job_id))
        routes[capability] = dict(job=job_id, family=family, command=tool_command,
                                  cwd=str(run), environment={})
        job = dict(id=job_id, family=family, task_id=row['task_id'],
                   category=row['category'], condition='latest_public_original_tools_one_pass',
                   repeat=1, answer_mode='text', numerical_threads=1,
                   prompt_file=str(prompt.relative_to(run)),
                   workspace=str(workspace.relative_to(run)))
        one_plan = run / 'plans' / (job_id + '.json')
        native_job = dict(job, prompt_file=str(prompt), workspace=str(workspace))
        one_plan.write_text(json.dumps(dict(jobs=[native_job]), ensure_ascii=False, indent=2) + '\n')
        mounts = run / 'plans' / (job_id + '_mounts.json')
        mounts.write_text(json.dumps(dict(mounts=[dict(
            source=str(run / 'ipc/fifo' / job_id), destination='/toolbridge')]), indent=2) + '\n')
        jobs.append(dict(**job, plan=str(one_plan), mounts=str(mounts)))
        for p in [prompt, workspace / 'inputs/question.txt', workspace / 'inputs/tool_connection.json',
                  workspace / 'tool_client.py', workspace / 'readonly_inputs.json', one_plan, mounts]:
            immutable.append(dict(path=str(p), sha256=digest(p)))
    # Predefined round-robin by category. This is scheduling, not score-driven
    # case selection. All75questions remain in the fixed denominator.
    groups = {}
    for job in jobs:
        groups.setdefault((job['family'], job['category']), []).append(job)
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.pop(0))
    broker = dict(transport='fifo', fifo_root=str(run / 'ipc/fifo'),
                  socket=str(run / 'ipc/broker.sock'), traces=str(run / 'traces'), routes=routes)
    (run / 'BROKER187.json').write_text(json.dumps(broker, indent=2) + '\n')
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), primary_passes=1,
        model='deepseek-flash', cli_version='0.163.0',
        cli='/data/yyf-zyq/PHM_20261010/20261015T215024Z_independent95/runtime/codex0163/package_files/package/vendor/x86_64-unknown-linux-musl/bin/codex',
        cli_sha256='399ff69c431c066e40de318fba45c2b07f45a76b19939d247ee4828fc70c4215',
        published_questions=78, forge_questions=75, asset_available_questions=3,
        asset_main_profile_denominator=232, asset_main_missing_question_manifest_gold=229,
        asset_HF152='Separate official legacy contract; no automatic substitution for232main suite.',
        phmforge_v3_99='Not reproduced by75-question public source;24battery extension not published in checked branches.',
        subject_timeout_seconds=1200, original_tool_timeout_seconds=180,
        max_concurrency=8, max_new_primary_rolling24h=120,
        numerical_threads=1, storage_budget_GiB=8,
        total_subject_wall_budget_hours=26, gpu=0, api_prices='unverified; no fabricated currency estimate',
        token_budget='native CLI context and1200s wall limit; actual token usage retained',
        service_model_revision='unreported unless returned in native events',
        interface='Codex CLI calls original tools through isolated JSON transport; '
                  'PHMForge uses same BaseTools as original automated runner; AssetOps uses original FastMCP tool handlers and real CouchDB.',
        data_caveats='PHMForge published heuristics/stubs cannot measure physical prediction accuracy. '
                     'Non-CMAPSS CSV payloads and some requested operations may be absent; do not fabricate them. '
                     'Report tool/data coverage and invalidity separately from Agent errors.',
        scoring='After all executed questions terminal, invoke unchanged PHMForge evaluate_pass and '
                'unchanged AssetOps static_json; supplement with trace audit. '
                'No same-model LLM judge and no alteration of gold or success thresholds.',
        outcomes=['success', 'task_failure', 'timeout', 'execution_error', 'data_or_tool_unavailable',
                  'scoring_unavailable', 'not_run_missing_source'],
        no_scores_to_subjects=True, no_retries_for_bad_scientific_results=True,
        launch_gate='One actual original subject first; expand only after terminal status, original tool use '
                    'and isolation/transport checks. At most2 environment repairs, attempts retained.',
        analysis='Only after full first pass, combine version-specific evidence with all archived history; '
                 'separate source/scorer defects, information insufficiency, execution and PHM failures. '
                 'Candidate mechanisms require measured consequence, ordinary review and rivals; no methods yet.')
    (run / 'PROTOCOL187.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    (run / 'PLAN187.json').write_text(json.dumps(dict(jobs=ordered), ensure_ascii=False, indent=2) + '\n')
    for p in [run / 'BROKER187.json', run / 'PROTOCOL187.json', run / 'PLAN187.json',
              project / 'scripts/benchmark_tool_bridge.py', project / 'scripts/replay_diagnostic_cli.py',
              run / 'cli_config187.toml', run / 'model_catalog187.json']:
        immutable.append(dict(path=str(p), sha256=digest(p)))
    (run / 'INITIAL_FREEZE187.json').write_text(json.dumps(dict(files=immutable), indent=2) + '\n')
    print(json.dumps(dict(prepared=len(jobs), forge=75, asset_main=3, missing_asset_main=229,
                          run=str(run)), ensure_ascii=False))


if __name__ == '__main__':
    main()
