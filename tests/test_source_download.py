"""Offline download-resume contracts using a local HTTP fixture, not PHM data."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import socket
import tempfile
import threading
import unittest

from phmforge_bench.data import _download_ranges


DATA = bytes(range(256)) * 2048


class RangeHandler(BaseHTTPRequestHandler):
    offsets = []
    interrupt_once = False

    def do_GET(self):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if not match:
            self.send_error(400)
            return
        start, end = map(int, match.groups())
        type(self).offsets.append(start)
        block = DATA[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(DATA)}")
        self.send_header("Content-Length", str(len(block)))
        self.end_headers()
        if type(self).interrupt_once:
            type(self).interrupt_once = False
            self.wfile.write(block[:4096])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
        else:
            self.wfile.write(block)

    def log_message(self, *_):
        pass


class SourceDownload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/fixture"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def setUp(self):
        RangeHandler.offsets = []
        RangeHandler.interrupt_once = False

    def test_resumes_existing_partial_and_an_interrupted_response(self):
        with tempfile.TemporaryDirectory(prefix="phmforge-download-contract-") as folder:
            path = Path(folder) / "source.bin"
            partial = path.with_name(path.name + ".partial.resume")
            partial.write_bytes(DATA[:12000])
            RangeHandler.interrupt_once = True
            _download_ranges(self.url, path, hashlib.sha256(DATA).hexdigest(), len(DATA), chunk_size=32768)
            self.assertEqual(path.read_bytes(), DATA)
            self.assertFalse(partial.exists())
            self.assertEqual(RangeHandler.offsets[0], 12000)
            self.assertGreater(RangeHandler.offsets[1], RangeHandler.offsets[0])

    def test_corrupt_partial_is_never_published_as_verified(self):
        with tempfile.TemporaryDirectory(prefix="phmforge-download-contract-") as folder:
            path = Path(folder) / "source.bin"
            partial = path.with_name(path.name + ".partial.resume")
            partial.write_bytes(b"wrong" * 1000)
            with self.assertRaisesRegex(ValueError, "failed pinned"):
                _download_ranges(self.url, path, hashlib.sha256(DATA).hexdigest(), len(DATA), chunk_size=32768)
            self.assertFalse(path.exists())
            self.assertTrue(partial.exists())


if __name__ == "__main__":
    unittest.main()
