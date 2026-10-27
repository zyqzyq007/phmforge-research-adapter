"""Run the complete pinned official workflow through qualified PHM I/O adapters.

This launcher is a qualification baseline, not a proposed selection mechanism.
The run's frozen plan determines its model, wallclock, search and execution caps.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from mlestar_phm_adapter import load_official, make_executor, make_model, make_search, save


def utc():
    return datetime.now(timezone.utc).isoformat()


async def pipeline(root):
    from google.adk import Runner
    from google.adk.agents import LlmAgent
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    plan = json.loads((root / 'PLAN431.json').read_text())
    execute = make_executor(root / 'workspace', root / 'executions',
                            root / 'code/program_training_entry.py')
    executions = 0

    def bounded_execute(code_text, run_cwd, py_filepath, exec_timeout):
        nonlocal executions
        if executions >= plan['code_execution_limit']:
            raise RuntimeError('Frozen generated-code execution budget exhausted')
        executions += 1
        save(root / 'EXECUTION_COUNT431.json', dict(attempts=executions, utc=utc()))
        return execute(code_text, run_cwd, py_filepath,
                       min(exec_timeout, plan['code_execution_seconds']))

    model = make_model(root / 'ipc/model.sock')
    module, inventory = load_official(
        root / 'source/machine_learning_engineering',
        dict(agent_model='deepseek-flash', data_dir=str(root / 'tasks'),
             workspace_dir=str(root / 'workspace'), task_name=plan['task_name'],
             exec_timeout=plan['code_execution_seconds']),
        model, make_search(root / 'search_broker', limit=plan['search_limit'],
                           wait_seconds=plan['search_timeout_seconds']), bounded_execute)

    def configure_output(agent):
        if isinstance(agent, LlmAgent):
            if agent.generate_content_config is None:
                agent.generate_content_config = types.GenerateContentConfig()
            agent.generate_content_config.max_output_tokens = plan['model_output_per_request']
        for child in agent.sub_agents:
            configure_output(child)
    configure_output(module.root_agent)
    save(root / 'WORKFLOW_GRAPH431.json', inventory)
    # Enter the complete scientific workflow directly. The conversational
    # front-door is not needed to dispatch an already explicit task.
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name='mlestar431', user_id='research')
    runner = Runner(agent=module.mle_pipeline_agent, app_name='mlestar431',
                    session_service=sessions)
    authors, events = [], 0
    error = None
    try:
        with (root / 'EVENTS431.jsonl').open('w') as log:
            async for event in runner.run_async(user_id='research', session_id=session.id,
                    new_message=types.Content(role='user', parts=[types.Part(text=(
                        'Complete the provided PHM task through initialization, targeted '
                        'refinement, ensemble exploration and final submission. Use real '
                        'search when retrieving models. Follow the exact task data and '
                        'evaluation contract.'))])):
                log.write(event.model_dump_json(exclude_none=True) + '\n'); log.flush()
                events += 1
                if event.author not in authors:
                    authors.append(event.author)
                current = await sessions.get_session(app_name='mlestar431', user_id='research',
                                                     session_id=session.id)
                save(root / 'SESSION_STATE431.json', current.state)
                save(root / 'PIPELINE_PROGRESS431.json', dict(
                    utc=utc(), events=events, last_author=event.author,
                    authors=authors, code_execution_attempts=executions))
    except BaseException as failure:
        error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
        raise
    finally:
        current = await sessions.get_session(app_name='mlestar431', user_id='research',
                                             session_id=session.id)
        save(root / 'SESSION_STATE431.json', current.state)
        save(root / 'PIPELINE_RESULT431.json', dict(
            utc=utc(), normal=error is None, error=error, events=events,
            authors=authors, code_execution_attempts=executions,
            hidden_final_access=False, is_method_comparison=False))


def control(root):
    plan = json.loads((root / 'PLAN431.json').read_text())
    env = dict(PATH='/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=f"{root}/code:{plan['dependency_root']}:/srv/yyf-zyq/PHM_20261010/_shared/verified_benchmarks246_v4/deps",
               TMPDIR=str(root / 'runtime/tmp'), TMP=str(root / 'runtime/tmp'),
               TEMP=str(root / 'runtime/tmp'), MPLCONFIGDIR=str(root / 'runtime/mpl'),
               XDG_CACHE_HOME=str(root / 'clients-cache'), OTEL_SDK_DISABLED='true')
    result, child = {}, None
    started = time.monotonic()
    with (root / 'logs/gateway.log').open('wb') as gout, (root / 'logs/pipeline.log').open('wb') as out:
        gateway = subprocess.Popen([sys.executable, '-B', str(root / 'code/isolated_model_gateway.py'),
                                    'serve', '--run', str(root)], env=env, stdout=gout, stderr=subprocess.STDOUT)
        try:
            for _ in range(200):
                if (root / 'MODEL_GATEWAY_READY.json').exists():
                    break
                if gateway.poll() is not None:
                    raise RuntimeError('Gateway exited before readiness')
                time.sleep(.1)
            assert (root / 'MODEL_GATEWAY_READY.json').exists()
            child = subprocess.Popen([sys.executable, '-B', str(root / 'code/run_mlestar_phm.py'),
                                      '--run', str(root), '--mode', 'pipeline'], env=env,
                                     stdout=out, stderr=subprocess.STDOUT)
            save(root / 'HANDLE431.json', dict(utc=utc(), controller=os.getpid(),
                                              gateway=gateway.pid, pipeline=child.pid))
            last_disk_check = 0
            while child.poll() is None:
                elapsed = time.monotonic() - started
                reason = 'whole_pipeline_wallclock' if elapsed >= plan['whole_pipeline_seconds'] else None
                if elapsed - last_disk_check >= 15:
                    last_disk_check = elapsed
                    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                    if size > plan['disk_bytes']:
                        reason = 'disk_budget'
                if reason:
                    result['budget_stop'] = reason
                    child.terminate(); child.wait(timeout=15)
                    break
                time.sleep(1)
            result['returncode'] = child.returncode
        except BaseException as failure:
            result.update(error_type=type(failure).__name__, error=str(failure))
            if child is not None and child.poll() is None:
                child.terminate(); child.wait(timeout=15)
        finally:
            # Let the external real-search service stop as soon as the runner
            # ends; gateway draining can take longer for a timed-out request.
            save(root / 'PIPELINE_STOPPED431.json', dict(utc=utc(), **result))
            (root / 'MODEL_GATEWAY_STOP').touch()
            try:
                gateway.wait(timeout=650)
            except subprocess.TimeoutExpired:
                gateway.terminate(); gateway.wait(timeout=10)
                result['gateway_forced_stop'] = True
            result.update(utc=utc(), gateway_returncode=gateway.returncode,
                          total_seconds=time.monotonic()-started)
            save(root / 'CONTROLLER_TERMINAL431.json', result)
    print(json.dumps(result))
    return result.get('returncode', 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--mode', choices=['control', 'pipeline'], required=True)
    args = parser.parse_args(); root = args.run.resolve()
    if args.mode == 'pipeline':
        asyncio.run(pipeline(root))
    else:
        raise SystemExit(control(root))
