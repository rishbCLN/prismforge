"""HazardMesh Demo Runner Script.
Launches the interactive Web Application on local HTTP server.
Usage:
    python scripts/run_demo.py [--port 8000] [--host 127.0.0.1]
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Launch HazardMesh Interactive Safety Intelligence Demo")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--reload", action="store_true", help="Auto reload on file change")
    args = parser.parse_args()

    print("=" * 70)
    print("      HAZARDMESH — CONTEXT-AWARE SAFETY INTELLIGENCE DEMO     ")
    print("=" * 70)
    print(f"Starting server at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    print("=" * 70)

    uvicorn.run("web.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
