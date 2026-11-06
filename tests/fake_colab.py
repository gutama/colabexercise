"""Test double for the `colab` CLI (google-colab-cli 0.7.5 command shapes).

Remote paths under /content map to $FAKE_COLAB_ROOT/content and `exec` runs code
locally. Like the real CLI, `exec` exits 0 even when the code raises -- which is the
behaviour the driver must not trust. FAKE_COLAB_UNAUTH=1 mimics a missing login.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ["FAKE_COLAB_ROOT"])
STATE = ROOT / "sessions.json"


def local(remote: str) -> Path:
    return ROOT / remote.lstrip("/")


def main() -> int:
    argv = sys.argv[1:]
    with (ROOT / "calls.log").open("a") as log:
        log.write(" ".join(argv) + "\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("-s", "--session")
    parser.add_argument("-f", "--file")
    parser.add_argument("--timeout")
    parser.add_argument("--gpu")
    parser.add_argument("--high-mem", action="store_true")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    sessions = json.loads(STATE.read_text()) if STATE.exists() else []

    if os.environ.get("FAKE_COLAB_UNAUTH"):
        print("To authorize colab-cli, visit this URL in any browser:\n"
              "  https://accounts.google.com/o/oauth2/auth?...\nAborted.")
        return 1
    if args.command == "sessions":
        print("\n".join(f"{name}  [CPU]" for name in sessions) or "No active sessions")
        return 0
    if args.command == "new":
        sessions.append(args.session)
        STATE.write_text(json.dumps(sessions))
        (ROOT / "content").mkdir(parents=True, exist_ok=True)
        return 0
    if args.command == "stop":
        STATE.write_text(json.dumps([s for s in sessions if s != args.session]))
        return 0
    if args.session not in sessions:
        print(f"Session '{args.session}' not found")
        return 1
    if args.command == "upload":
        source, target = args.paths
        if not local(target).parent.is_dir():
            print("HTTP 500")
            return 1
        shutil.copyfile(source, local(target))
        return 0
    if args.command == "download":
        source, target = args.paths
        shutil.copyfile(local(source), target)
        return 0
    if args.command == "exec":
        code = Path(args.file).read_text() if args.file else sys.stdin.read()
        env = dict(os.environ)
        for item in args.env:
            key, value = item.split("=", 1)
            env[key] = str(local(value)) if value.startswith("/content") else value
        subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT / "content")
        return 0
    print(f"unsupported command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
