#!/usr/bin/env python3
"""
Download ViT-B/16 checkpoints for MAE, MoCo v3, and CMAE into the
project's checkpoint directory.

Default outputs:
  checkpoints/pretrained/mae_timm_vit_base_patch16_224_mae.pth
  checkpoints/pretrained/moco_v3_vit_b_300ep.pth.tar
  checkpoints/pretrained/cmae_vit_base_p16_300e_in1k.pth
  checkpoints/pretrained/download_manifest.json

Usage:
  python -m src.utils.download_models
  python src/utils/download_models.py
  python -m src.utils.download_models --project-dir /path/to/project
  python -m src.utils.download_models --cmae-epochs 1600
  python -m src.utils.download_models --clone-repos
  python -m src.utils.download_models --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict


DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
MOCO_V3_VIT_B_300EP_URL = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
CMAE_VIT_B_300EP_URL = "https://cmae.s3.us-west-1.amazonaws.com/weight/cmae_vit-base-p16_32xb128-coslr-300e_in1k.pth"
CMAE_VIT_B_1600EP_URL = "https://cmae.s3.us-west-1.amazonaws.com/weight/cmae_vit-base-p16_32xb128-coslr-1600e_in1k.pth"

REPOS = {
    "moco-v3": "https://github.com/facebookresearch/moco-v3.git",
    "CMAE": "https://github.com/ZhichengHuang/CMAE.git",
}

MODEL_METADATA = {
    "mae_timm_vit_base_patch16_224": {
        "method": "MAE",
        "architecture": "ViT-B/16",
        "pretraining_dataset": "ImageNet-1K",
        "source": "https://huggingface.co/timm/vit_base_patch16_224.mae",
        "license": "CC-BY-NC-4.0",
    },
    "moco_v3_vit_b_300ep": {
        "method": "MoCo v3",
        "architecture": "ViT-B/16",
        "pretraining_dataset": "ImageNet-1K",
        "pretraining_epochs": 300,
        "source": "https://github.com/facebookresearch/moco-v3",
        "license": "CC-BY-NC-4.0",
    },
    "cmae_vit_base_p16": {
        "method": "CMAE",
        "architecture": "ViT-B/16",
        "pretraining_dataset": "ImageNet-1K",
        "source": "https://github.com/ZhichengHuang/CMAE",
    },
}


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


class Progress:
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(100, int(downloaded * 100 / total_size))
        if percent >= self.last_percent + 5 or percent == 100:
            self.last_percent = percent
            print(f"  {self.name}: {percent}%", flush=True)


def download_url(url: str, dest: Path, force: bool = False) -> Dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists")
        return {
            "path": str(dest),
            "url": url,
            "bytes": dest.stat().st_size,
            "sha256": sha256sum(dest),
            "status": "already_exists",
        }

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"[download] {url}\n  -> {dest}")
    try:
        urllib.request.urlretrieve(url, tmp, Progress(dest.name))
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    return {
        "path": str(dest),
        "url": url,
        "bytes": dest.stat().st_size,
        "sha256": sha256sum(dest),
        "status": "downloaded",
    }


def download_mae_timm(models_dir: Path, force: bool = False) -> Dict[str, Any]:
    """Download timm MAE through timm/HF Hub and save a local .pth checkpoint."""
    out = models_dir / "mae_timm_vit_base_patch16_224_mae.pth"
    if out.exists() and not force:
        print(f"[skip] {out} already exists")
        return {
            "path": str(out),
            "model_name": "hf_hub:timm/vit_base_patch16_224.mae",
            "bytes": out.stat().st_size,
            "sha256": sha256sum(out),
            "status": "already_exists",
        }

    # Keep all auto-downloaded cache files inside the checkpoint tree.
    os.environ.setdefault("HF_HOME", str(models_dir / "_cache" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(models_dir / "_cache" / "torch"))
    os.environ.setdefault("TIMM_CACHE_DIR", str(models_dir / "_cache" / "timm"))

    try:
        import torch  # type: ignore
        import timm  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Missing dependency. Install first, e.g.:\n"
            "  pip install torch timm huggingface_hub safetensors\n"
            f"Original error: {e}"
        )

    model_name = "hf_hub:timm/vit_base_patch16_224.mae"
    print(f"[timm] loading pretrained {model_name}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    checkpoint = {
        "model_name": model_name,
        "source": "https://huggingface.co/timm/vit_base_patch16_224.mae",
        "state_dict": model.state_dict(),
    }
    print(f"[save] {out}")
    torch.save(checkpoint, out)

    return {
        "path": str(out),
        "model_name": model_name,
        "source": checkpoint["source"],
        "bytes": out.stat().st_size,
        "sha256": sha256sum(out),
        "status": "downloaded",
    }


def planned_entry(path: Path, url: str | None = None, **metadata: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"path": str(path), "status": "planned"}
    if url is not None:
        entry["url"] = url
    entry.update(metadata)
    return entry


def clone_repo(name: str, url: str, dest: Path, force: bool = False) -> Dict[str, Any]:
    if dest.exists() and not force:
        print(f"[skip] repo {name} already exists at {dest}")
        return {"name": name, "url": url, "path": str(dest), "status": "already_exists"}
    if dest.exists() and force:
        shutil.rmtree(dest)
    print(f"[clone] {url}\n  -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    commit = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    return {"name": name, "url": url, "path": str(dest), "commit": commit, "status": "cloned"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MAE/timm, MoCo v3, and CMAE ViT-B/16 checkpoints.")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR, help="Project root. Default: repository root inferred from this file.")
    parser.add_argument("--models-subdir", default="checkpoints/pretrained", help="Subdirectory for checkpoints. Default: checkpoints/pretrained.")
    parser.add_argument("--cmae-epochs", choices=["300", "1600"], default="300", help="CMAE pretraining checkpoint to download.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--clone-repos", action="store_true", help="Also clone official MoCo v3 and CMAE repos under the checkpoint directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without creating files or downloading checkpoints.")
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    models_dir = project_dir / args.models_subdir

    manifest: Dict[str, Any] = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_dir": str(project_dir),
        "models_dir": str(models_dir),
        "models": {},
        "repos": {},
    }

    cmae_url = CMAE_VIT_B_300EP_URL if args.cmae_epochs == "300" else CMAE_VIT_B_1600EP_URL
    cmae_name = f"cmae_vit_base_p16_{args.cmae_epochs}e_in1k.pth"

    if args.dry_run:
        manifest["models"]["mae_timm_vit_base_patch16_224"] = planned_entry(
            models_dir / "mae_timm_vit_base_patch16_224_mae.pth",
            MODEL_METADATA["mae_timm_vit_base_patch16_224"]["source"],
            **MODEL_METADATA["mae_timm_vit_base_patch16_224"],
        )
        manifest["models"]["moco_v3_vit_b_300ep"] = planned_entry(
            models_dir / "moco_v3_vit_b_300ep.pth.tar",
            MOCO_V3_VIT_B_300EP_URL,
            **MODEL_METADATA["moco_v3_vit_b_300ep"],
        )
        manifest["models"][f"cmae_vit_base_p16_{args.cmae_epochs}e"] = planned_entry(
            models_dir / cmae_name,
            cmae_url,
            **MODEL_METADATA["cmae_vit_base_p16"],
            pretraining_epochs=int(args.cmae_epochs),
        )
        if args.clone_repos:
            repos_dir = models_dir / "repos"
            for name, url in REPOS.items():
                manifest["repos"][name] = planned_entry(repos_dir / name, url, name=name)
        print(json.dumps(manifest, indent=2, ensure_ascii=True))
        return 0

    models_dir.mkdir(parents=True, exist_ok=True)

    manifest["models"]["mae_timm_vit_base_patch16_224"] = download_mae_timm(models_dir, force=args.force)
    manifest["models"]["mae_timm_vit_base_patch16_224"].update(MODEL_METADATA["mae_timm_vit_base_patch16_224"])

    manifest["models"]["moco_v3_vit_b_300ep"] = download_url(
        MOCO_V3_VIT_B_300EP_URL,
        models_dir / "moco_v3_vit_b_300ep.pth.tar",
        force=args.force,
    )
    manifest["models"]["moco_v3_vit_b_300ep"].update(MODEL_METADATA["moco_v3_vit_b_300ep"])

    manifest["models"][f"cmae_vit_base_p16_{args.cmae_epochs}e"] = download_url(
        cmae_url,
        models_dir / cmae_name,
        force=args.force,
    )
    manifest["models"][f"cmae_vit_base_p16_{args.cmae_epochs}e"].update(
        MODEL_METADATA["cmae_vit_base_p16"] | {"pretraining_epochs": int(args.cmae_epochs)}
    )

    if args.clone_repos:
        repos_dir = models_dir / "repos"
        for name, url in REPOS.items():
            manifest["repos"][name] = clone_repo(name, url, repos_dir / name, force=args.force)

    manifest_path = models_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
