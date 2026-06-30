"""
Downloads the NEU Surface Defect dataset (6 classes, ~1800 grayscale images
of hot-rolled steel strip defects) from a public GitHub mirror, and organizes
it into data/raw/<class>/*.jpg ready for src/prepare_data.py.

Original dataset: K. Song and Y. Yan, Northeastern University (NEU).
Mirror used here: github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection
(Used for academic / non-commercial purposes only, per original dataset terms.)

Usage:
    python src/download_data.py
"""
import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MIRROR_URL = "https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection.git"
CLONE_DIR = ROOT / "data" / "_neu_mirror_tmp"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def _force_remove_readonly(func, path, exc_info):
    """Handler for shutil.rmtree: clears read-only/locked bits (common with
    git/git-lfs files on Windows) and retries the removal."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _rmtree_safe(path: Path):
    if path.exists():
        shutil.rmtree(path, onerror=_force_remove_readonly)


def main():
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"{RAW_DIR} already has data. Delete it first if you want to re-download.")
        return

    print(f"Cloning dataset mirror from {MIRROR_URL} ...")
    _rmtree_safe(CLONE_DIR)

    # The mirror repo also tracks a large unrelated model file via Git LFS
    # (Models/model_fpn_inceptionv4.pth, ~524MB). We don't need it, and the
    # mirror's shared LFS bandwidth budget is often exhausted by other users.
    # Two layers of protection against ever touching that file:
    #   1. GIT_LFS_SKIP_SMUDGE=1 -> don't download LFS file contents on checkout
    #   2. sparse-checkout -> only check out the IMAGES/ folder we actually need
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "init"], cwd=CLONE_DIR, check=True, env=env,
                        capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", MIRROR_URL], cwd=CLONE_DIR,
                        check=True, env=env, capture_output=True)
        subprocess.run(["git", "config", "core.sparseCheckout", "true"], cwd=CLONE_DIR,
                        check=True, env=env, capture_output=True)
        sparse_file = CLONE_DIR / ".git" / "info" / "sparse-checkout"
        sparse_file.parent.mkdir(parents=True, exist_ok=True)
        sparse_file.write_text("IMAGES/*\n")
        subprocess.run(["git", "fetch", "--depth", "1", "origin"], cwd=CLONE_DIR,
                        check=True, env=env, capture_output=True)
        subprocess.run(["git", "checkout", "FETCH_HEAD"], cwd=CLONE_DIR,
                        check=True, env=env, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore") if isinstance(e.stderr, bytes) else e.stderr
        print(f"git command failed: {stderr}")
        raise

    images_dir = CLONE_DIR / "IMAGES"
    if not images_dir.exists():
        raise RuntimeError(f"Expected IMAGES folder not found at {images_dir}. "
                            f"The mirror repo structure may have changed.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        files = list(images_dir.glob(f"{cls}_*.jpg"))
        for f in files:
            shutil.copy(f, cls_dir / f.name)
        print(f"{cls}: {len(files)} images")

    _rmtree_safe(CLONE_DIR)
    print(f"\nDone. Raw data ready at: {RAW_DIR}")
    print("Next step: python src/prepare_data.py")


if __name__ == "__main__":
    main()