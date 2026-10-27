"""Actual MCP or JSON-lines entry point for explicitly verified benchmark profiles."""
import argparse
import asyncio
import contextlib
import importlib
import json
from pathlib import Path
import sys
import time


async def serve(args):
    if args.family == "forge":
        from verified_phmforge import VerifiedForge
        backend = VerifiedForge(args.data, args.output)
        schemas = json.loads((Path(args.data) / "tool_catalog.json").read_text())
        async def invoke(name, arguments):
            return backend.call(name, arguments)
    else:
        from qualify_asset_runtime import json_value
        names = ["tsfm"] if args.family == "asset_tsfm" else ["wo", "iot", "tsfm", "utilities", "fmsr", "vibration"]
        servers = {n: importlib.import_module("servers."+n+".main").mcp for n in names}
        schemas = {}
        for prefix, mcp in servers.items():
            schemas.update({prefix+"."+t.name: t.model_dump(mode="json") for t in await mcp.list_tools()})
        async def invoke(name, arguments):
            prefix, tool = name.split(".", 1)
            return json_value(await servers[prefix].call_tool(tool, arguments))
    async def recorded(name, arguments):
        start = time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):
            result = await invoke(name, arguments)
        record = dict(tool=name, arguments=arguments, result=result, elapsed_seconds=time.monotonic()-start)
        with (Path(args.output) / "tool_calls.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return result
    if args.transport == "jsonl":
        for line in sys.stdin:
            start = time.monotonic()
            try:
                req = json.loads(line)
                if req["tool"] in ["list_tools", "full_catalog"]:
                    result = schemas
                elif req["tool"] == "tool_schema":
                    result = schemas[req["args"]["name"]]
                else:
                    result = await recorded(req["tool"], req.get("args", {}))
                out = dict(transport_ok=True, result=result)
            except Exception as exc:
                out = dict(transport_ok=False, error_type=type(exc).__name__, error=str(exc))
            out["elapsed_seconds"] = time.monotonic()-start
            print(json.dumps(out, default=str), flush=True)
    else:
        import mcp.types as types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        server = Server("verified-"+args.family)
        @server.list_tools()
        async def tools():
            return [types.Tool(name=n, description=v["description"], inputSchema=v["inputSchema"])
                    for n, v in schemas.items()]
        @server.call_tool()
        async def call_tool(name, arguments):
            result = await recorded(name, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["forge", "asset", "asset_tsfm"], required=True)
    p.add_argument("--data", default="/app")
    p.add_argument("--output", default="/output")
    p.add_argument("--transport", choices=["jsonl", "mcp"], default="mcp")
    args = p.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    asyncio.run(serve(args))
