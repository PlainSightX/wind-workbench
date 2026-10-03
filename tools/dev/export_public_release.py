"""按审阅清单导出候选，不复制私有历史/runtime；只创建新目录。"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil


def public_files(root):
    config = json.loads((root / "docs/public-release-files.json").read_text("utf-8-sig"))
    exclude = set(config["exclude_parts"])
    sources = {name: name for name in config["files"]}
    for directory in config["directories"]:
        for path in (root / directory).rglob("*"):
            name = path.relative_to(root).as_posix()
            if path.is_file() and not (set(path.relative_to(root).parts) & exclude):
                sources[name] = name
    sources.update(config["replace"])
    for target, source in sources.items():
        for name in (target, source):
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or set(path.parts) & exclude:
                raise ValueError("public_path_invalid:" + name)
        path = root / source
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("public_source_invalid:" + source)
    return sources


def export(root, destination):
    root, destination = root.resolve(), destination.absolute()
    if destination.exists():
        raise FileExistsError(destination)
    sources = public_files(root)
    manifest = {}
    # 所有输入先审查，再创建候选；凭据/机器路径不得混入文本交付。
    for name, source in sorted(sources.items()):
        payload = (root / source).read_bytes()
        if Path(name).suffix not in {".png", ".webm", ".ico"}:
            content = payload.decode("utf-8-sig")
            if re.search(r"sk-[a-zA-Z0-9]{24,}|C:[/\\]Users[/\\]fuxia|D:[/\\]CodexWorkspace", content):
                raise ValueError("public_private_content:" + source)
        manifest[name] = {"source": source, "sha256": sha256(payload).hexdigest(), "bytes": len(payload)}
    destination.mkdir(parents=True)
    for name, source in sorted(sources.items()):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / source, path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = export(args.root, args.destination)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(manifest), "bytes": sum(r["bytes"] for r in manifest.values())}))


if __name__ == "__main__":
    main()
