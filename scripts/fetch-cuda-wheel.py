"""Optional Windows download accelerator. Official index + SHA256, resumable ranges.

Does not install packages or alter network settings. Each range is validated before
appending. This exists because the CUDA wheel is 2.87GB and single-stream downloads
can be slow on this connection.
"""
import concurrent.futures
import argparse
import hashlib
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", choices=["official", "sjtu"], default="official")
    args = parser.parse_args()
    index = "https://download.pytorch.org/whl/cu128/torch/"
    content = urllib.request.urlopen(index, timeout=30).read().decode()
    links = [html.unescape(x) for x in re.findall(r'href="([^"]+)"', content)]
    links = [x for x in links if "torch-2.10.0+cu128-cp312-cp312-win_amd64.whl" in urllib.parse.unquote(x)]
    if len(links) != 1:
        raise RuntimeError("Expected exactly one official Windows CPython 3.12 CUDA wheel")
    full = urllib.parse.urljoin(index, links[0])
    url, fragment = urllib.parse.urldefrag(full)
    expected = urllib.parse.parse_qs(fragment)["sha256"][0]
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise RuntimeError("Server does not support validated ranges")
        size = int(response.headers["Content-Range"].split("/")[1])
    transfer_url = url if args.mirror == "official" else "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/" + url.rsplit("/", 1)[1]
    folder = ROOT / ".cache" / "wheels"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "torch-2.10.0+cu128-cp312-cp312-win_amd64.whl"
    parts = folder / f"torch-{expected[:12]}-parts"
    parts.mkdir(exist_ok=True)
    chunk_size = 64 * 1024**2
    started = time.monotonic()

    def fetch(i):
        start, end = i*chunk_size, min(size, (i+1)*chunk_size)-1
        path = parts / f"{i:03}.part"
        target_size = end-start+1
        for retry in range(5):
            existing = path.stat().st_size if path.exists() else 0
            if existing == target_size:
                return
            if existing > target_size:
                raise RuntimeError("Range cache is oversized")
            actual = start+existing
            ranged_url = transfer_url + "?segment=" + str(i)
            req = urllib.request.Request(ranged_url, headers={"Range": f"bytes={actual}-{end}"})
            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {actual}-{end}/{size}":
                        raise RuntimeError("Server returned the wrong range")
                    with path.open("ab") as f:
                        while True:
                            block = response.read(1024*1024)
                            if not block:
                                break
                            f.write(block)
                if path.stat().st_size == target_size:
                    print(f"range {i+1}/{(size+chunk_size-1)//chunk_size} complete; elapsed={time.monotonic()-started:.1f}s", flush=True)
                    return
            except Exception:
                if retry == 4:
                    raise
                time.sleep(min(2**retry, 10))
        raise RuntimeError("Incomplete range")

    count = (size+chunk_size-1)//chunk_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, range(count)))
    checksum = hashlib.sha256()
    temp = output.with_suffix(".assembling")
    with temp.open("wb") as f:
        for i in range(count):
            with (parts / f"{i:03}.part").open("rb") as source:
                for block in iter(lambda: source.read(4*1024**2), b""):
                    f.write(block)
                    checksum.update(block)
    if checksum.hexdigest() != expected or temp.stat().st_size != size:
        raise RuntimeError("Official wheel SHA256 validation failed")
    os.replace(temp, output)
    (folder / "torch_source.json").write_text(json.dumps({"official_url": url, "transfer_url": transfer_url, "sha256": expected, "bytes": size}, indent=2), encoding="utf-8")
    print(f"VERIFIED {output}", flush=True)


if __name__ == "__main__":
    main()
