"""Start the Gradio app, capture frames with headless Chromium (Playwright via
node, tools/capture_frames.js) and assemble assets/demo.gif.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
PORT = 7861


def main():
    proc = subprocess.Popen([sys.executable, str(ROOT / "app.py")], env=dict(os.environ, PORT=str(PORT)),
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = Path(tempfile.mkdtemp())
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}", timeout=1)
                break
            except Exception:
                time.sleep(1)
        imgs = [str(p) for p in sorted((ROOT / "assets" / "examples").glob("*.jpg"))[:3]]
        subprocess.run(["node", str(ROOT / "tools" / "capture_frames.js"), str(PORT), str(out), *imgs],
                       check=True, env=dict(os.environ, NODE_PATH="/opt/node-tools/node_modules"))
        files = sorted(out.glob("*.png"))
        frames = [Image.open(f).convert("RGB") for f in files]
        durs = [int(f.stem.split("_")[1]) for f in files]
        frames = [f.resize((960, int(960 * f.height / f.width))) for f in frames]
        frames[0].save(ROOT / "assets" / "demo.gif", save_all=True, append_images=frames[1:],
                       duration=durs, loop=0, optimize=True)
        print("frames:", len(frames), "seconds:", sum(durs) / 1000)
    finally:
        proc.terminate()
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
