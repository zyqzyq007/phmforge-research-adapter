"""Offline, correlated file responses for original benchmark tool requests."""
import errno
import fcntl
import json
import os
from pathlib import Path
import select
import sys
import time
import uuid


def fifo_request(directory, route, tool, arguments, timeout=200, request_root=None):
    directory = Path(directory)
    request_id = uuid.uuid4().hex
    deadline = time.monotonic() + timeout
    request = dict(route=route, request_id=request_id, tool=tool, args=arguments)
    request_root = Path(request_root or '/app/.runtime/tool_requests')
    request_root.mkdir(parents=True, exist_ok=True)
    body = request_root / ('request_' + request_id + '.json')
    temporary = body.with_suffix('.new')
    temporary.write_text(json.dumps(request) + '\n')
    os.replace(temporary, body)
    # Only a small atomic notification enters the FIFO. Killing a client during
    # a large request cannot leave a partial JSON line for the next client.
    payload = (json.dumps(dict(route=route, request_id=request_id)) + '\n').encode()
    fd = None
    try:
        while fd is None:
            try:
                fd = os.open(directory / 'request.fifo', os.O_WRONLY | os.O_NONBLOCK)
            except OSError as exc:
                if exc.errno not in (errno.ENXIO, errno.ENOENT):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError('Tool broker has no request reader') from exc
                time.sleep(0.05)
        if len(payload) > os.fpathconf(fd, 'PC_PIPE_BUF'):
            raise ValueError('Tool notification exceeds atomic FIFO write size')
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Tool request write exceeded deadline')
            _, writable, _ = select.select([], [fd], [], min(remaining, 1))
            if writable:
                try:
                    written = os.write(fd, payload)
                    if written != len(payload):
                        raise RuntimeError('Partial tool notification write')
                    break
                except BlockingIOError:
                    pass
    finally:
        if fd is not None:
            os.close(fd)
    response = directory / ('response_' + request_id + '.json')
    while not response.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Tool response exceeded deadline: ' + request_id)
        time.sleep(0.02)
    envelope = json.loads(response.read_text())
    if envelope['request_id'] != request_id:
        raise RuntimeError('Tool response request identifier mismatch')
    return envelope['response']


def main():
    cfg = json.loads(Path('/app/inputs/tool_connection.json').read_text())
    name = sys.argv[1] if len(sys.argv) > 1 else 'list_tools'
    arguments = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    if len(sys.argv) > 3:
        raise SystemExit('Use: python3 tool_client.py TOOL JSON_OBJECT')
    deadline = time.monotonic() + 200
    with Path('/app/.runtime/tool_bridge.lock').open('a+') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Tool workspace lock exceeded deadline')
                time.sleep(0.05)
        data = fifo_request('/toolbridge', cfg['route'], name, arguments,
                            timeout=max(0, deadline - time.monotonic()))
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
