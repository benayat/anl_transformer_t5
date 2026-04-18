
#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys, json
from pathlib import Path

FIXED = json.loads('{"recipe": "multitask", "stage": "warmstart"}')
TARGET = Path(__file__).with_name("train.py")

def main():
    cmd = [sys.executable, str(TARGET)]
    for key, value in FIXED.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    cmd.extend(sys.argv[1:])
    raise SystemExit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
