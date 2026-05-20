"""
Download openBMI, HGD, and CBCIC raw .mat files for MLEMSDA.

Usage (from repo root):
    python scripts/download_datasets.py --dataset cbcic
    python scripts/download_datasets.py --dataset hgd --hgd-method gin
    python scripts/download_datasets.py --dataset hgd --hgd-method http
    python scripts/download_datasets.py --dataset hgd --verify
    python scripts/download_datasets.py --dataset openbmi --openbmi-subjects 1-2

HGD (High-Gamma Dataset) official sources:
  - Repo: https://github.com/robintibor/high-gamma-dataset
  - GIN:  https://gin.g-node.org/robintibor/high-gamma-dataset
  - Load: braindecode.datasets.bbci.BBCIDataset(filename='train/1.mat').load()

Expected layout for MLEMSDA:
  MLEMSDA/process_cbcic_hgd_bmi/data/hgd/train/1.mat ... 14.mat
  MLEMSDA/process_cbcic_hgd_bmi/data/hgd/test/1.mat  ... 14.mat
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "MLEMSDA" / "process_cbcic_hgd_bmi" / "data"

OPENBMI_BASE = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
    "10.5524/100001_101000/100542"
)
# GIN raw file endpoint (HDF5 .mat)
HGD_BASE = "https://web.gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"
CBCIC_BASE = (
    "https://raw.githubusercontent.com/5anirban9/"
    "Clinical-Brain-Computer-Interfaces-Challenge-WCCI-2020-Glasgow/master"
)


def download_file(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"  get: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "MLEMSDA-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0) or 0)
                done = 0
                chunk = 1024 * 1024
                with open(tmp, "wb") as f:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        if total and done % (10 * chunk) < chunk:
                            pct = 100.0 * done / total
                            print(f"    {done}/{total} ({pct:.1f}%)", flush=True)
            tmp.replace(dest)
            print(f"  saved: {dest} ({dest.stat().st_size} bytes)")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  attempt {attempt}/{retries} failed: {exc}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def download_openbmi(subjects: range | None = None, sessions: tuple[int, ...] = (1, 2)) -> None:
    out_dir = DATA_ROOT / "openBMI"
    subjects = subjects or range(1, 55)
    for session in sessions:
        for subject in subjects:
            rel = f"session{session}/s{subject}/sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
            url = f"{OPENBMI_BASE}/{rel}"
            dest = out_dir / f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
            download_file(url, dest)


def download_hgd_http(subjects: range | None = None) -> None:
    """Direct HTTP from GIN raw URLs (may fail on some networks / SSL)."""
    subjects = subjects or range(1, 15)
    for split in ("train", "test"):
        out_dir = DATA_ROOT / "hgd" / split
        for sid in subjects:
            url = f"{HGD_BASE}/{split}/{sid}.mat"
            download_file(url, out_dir / f"{sid}.mat")


def download_hgd_gin(dest_root: Path | None = None) -> None:
    """
    Official GIN workflow (recommended by high-gamma-dataset README):

        pip install gin-client
        gin login          # once, browser auth
        gin get robintibor/high-gamma-dataset
        cd high-gamma-dataset
        gin download --content

    Then copy data/train/*.mat and data/test/*.mat into MLEMSDA data folder.
    """
    gin = shutil.which("gin")
    if gin is None:
        print("gin CLI not found. Install with: pip install gin-client")
        print("Then run the commands above manually, or use --hgd-method http.")
        raise RuntimeError("gin executable not on PATH")

    work = dest_root or (ROOT / "_downloads" / "high-gamma-dataset")
    work = work.resolve()
    if not (work / ".git").exists() and not (work / "data").exists():
        print(f"=== gin get -> {work} ===")
        work.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([gin, "get", "robintibor/high-gamma-dataset", str(work)], check=True)

    repo = work / "high-gamma-dataset" if (work / "high-gamma-dataset").exists() else work
    print(f"=== gin download --content in {repo} ===")
    subprocess.run([gin, "download", "--content"], cwd=repo, check=True)

    for split in ("train", "test"):
        src = repo / "data" / split
        dst = DATA_ROOT / "hgd" / split
        if not src.is_dir():
            raise RuntimeError(f"Missing after gin download: {src}")
        dst.mkdir(parents=True, exist_ok=True)
        for mat in sorted(src.glob("*.mat")):
            target = dst / mat.name
            if target.exists() and target.stat().st_size == mat.stat().st_size:
                print(f"  skip (exists): {target.name}")
                continue
            print(f"  copy: {mat.name}")
            shutil.copy2(mat, target)


def verify_hgd() -> None:
    """Verify .mat files using braindecode BBCIDataset (official loader)."""
    try:
        from braindecode.datasets.bbci import BBCIDataset
    except ImportError as exc:
        raise RuntimeError("pip install braindecode==0.5  # then retry --verify") from exc

    for split in ("train", "test"):
        sample = DATA_ROOT / "hgd" / split / "1.mat"
        if not sample.exists():
            raise FileNotFoundError(f"Missing: {sample}")
        print(f"  loading {sample} ...")
        cnt = BBCIDataset(filename=str(sample), load_sensor_names=None).load()
        print(f"    OK  channels={len(cnt.ch_names)}  duration={cnt.times[-1]:.1f}s")


def print_hgd_manual_help() -> None:
    print(
        """
HGD manual download (if gin/http both fail):
  1. Open https://gin.g-node.org/robintibor/high-gamma-dataset/src/master/data
  2. Enter folders train/ and test/
  3. Download 1.mat ... 14.mat for each folder
  4. Place files under:
       MLEMSDA/process_cbcic_hgd_bmi/data/hgd/train/
       MLEMSDA/process_cbcic_hgd_bmi/data/hgd/test/
  5. Verify: python scripts/download_datasets.py --dataset hgd --verify

After download, load in Python (official example):
  from braindecode.datasets.bbci import BBCIDataset
  cnt = BBCIDataset(filename='./train/1.mat', load_sensor_names=None).load()
"""
    )


def download_cbcic(participants: range | None = None) -> None:
    out_dir = DATA_ROOT / "cbcic"
    participants = participants or range(1, 11)
    for pid in participants:
        for suffix in ("E", "T"):
            name = f"parsed_P{pid:02d}{suffix}.mat"
            url = f"{CBCIC_BASE}/{name}"
            download_file(url, out_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MLEMSDA raw datasets")
    parser.add_argument(
        "--dataset",
        choices=("all", "openbmi", "hgd", "cbcic"),
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--openbmi-subjects",
        type=str,
        default="1-54",
        help="Subject range, e.g. 1-54 or 1-5 for a quick test",
    )
    parser.add_argument(
        "--hgd-method",
        choices=("gin", "http", "help"),
        default="gin",
        help="HGD download: gin (official CLI), http (direct URL), help (instructions only)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After HGD download, verify train/1.mat with braindecode BBCIDataset",
    )
    args = parser.parse_args()

    def parse_range(spec: str) -> range:
        a, b = spec.split("-", 1)
        return range(int(a), int(b) + 1)

    try:
        if args.dataset in ("all", "cbcic"):
            print("=== CBCIC (P01–P10) ===")
            download_cbcic()
        if args.dataset in ("all", "hgd"):
            if args.hgd_method == "help":
                print_hgd_manual_help()
            elif args.hgd_method == "gin":
                print("=== HGD via GIN CLI (official) ===")
                download_hgd_gin()
            else:
                print("=== HGD via HTTP (GIN raw URLs) ===")
                download_hgd_http()
            if args.verify:
                print("=== HGD verify (braindecode) ===")
                verify_hgd()
        if args.dataset in ("all", "openbmi"):
            print("=== openBMI (54 subjects × 2 sessions) ===")
            print("  Warning: ~10+ GB total; use --openbmi-subjects 1-2 to test first.")
            download_openbmi(subjects=parse_range(args.openbmi_subjects))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nDone. Data root:", DATA_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
