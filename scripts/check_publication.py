"""Check the explicit public file set without staging, deleting or uploading files."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = re.compile(r"mats|mech.interp.context|conversation|admissions", re.I)
SECRET = re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bhf_[A-Za-z0-9]{30,}\b|\bsk-[A-Za-z0-9]{32,}\b")


def check(root):
    manifest = json.loads((root / "publication_files.json").read_text())
    files = manifest["files"]
    if len(files) != len(set(files)):
        raise ValueError("Duplicate publication entries")
    total = 0
    for name in files:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or PRIVATE.search(name):
            raise ValueError(f"Unsafe/private publication path: {name}")
        path = root / rel
        if any(p.is_symlink() for p in [path, *path.parents] if p != root and root in p.parents):
            raise ValueError(f"Symlink in publication path: {name}")
        raw = path.read_bytes()
        if len(raw) > 2_000_000:
            raise ValueError(f"Unexpected large file: {name}")
        total += len(raw)
        text = raw.decode("utf-8")
        if SECRET.search(text):
            raise ValueError(f"Potential credential in {name}; inspect locally")
        if path.suffix == ".py":
            ast.parse(text, filename=name)
    if total > 5_000_000:
        raise ValueError("Publication source budget exceeded (5 MB)")
    if (root / ".git").exists():
        result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                cwd=root, check=True, capture_output=True)
        visible = set(result.stdout.decode().strip("\0").split("\0")) - {""}
        if visible != set(files):
            raise ValueError(f"Git/manifest mismatch: extra={sorted(visible-set(files))}, missing={sorted(set(files)-visible)}")
    return len(files), total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    n, size = check(args.root.resolve())
    print(f"Publication boundary OK: {n} files, {size:,} bytes. Nothing staged or uploaded.")
    print("Pattern checks are not a comprehensive secret, copyright or provenance review.")


if __name__ == "__main__":
    main()
