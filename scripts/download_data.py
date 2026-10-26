'下载固定版本的 UD English-EWT r2.15 数据及官方评估脚本。'
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "4dc8e10cf32352e11ab2c46e024b19853b91546e"
BASE = f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{COMMIT}"


def download():
    files = {f"data/raw/en_ewt-ud-{split}.conllu": f"{BASE}/en_ewt-ud-{split}.conllu"
             for split in ("train", "dev", "test")}
    files.update({"data/raw/LICENSE.txt": f"{BASE}/LICENSE.txt",
                  "data/raw/README_treebank.md": f"{BASE}/README.md",
                  "vendor/conll18_ud_eval.py": "https://universaldependencies.org/conll18/conll18_ud_eval.py"})
    manifest_path = ROOT / "data/manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {"release": "UD 2.15", "treebank": "English-EWT", "commit": COMMIT, "files": {}}
    for relative, url in files.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urlopen(url, timeout=90) as response:
                content = response.read()
            path.write_bytes(content)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = existing.get("files", {}).get(relative, {}).get("sha256")
        if expected and digest != expected:
            raise ValueError(f"文件摘要不一致： {relative}")
        manifest["files"][relative] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
        print(relative, path.stat().st_size, digest[:12], flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    download()
