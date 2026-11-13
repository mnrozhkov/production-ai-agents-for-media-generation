"""Run a Sana Diffusers pipeline sequentially and write assets, metrics.json, and trace.jsonl."""
import argparse
import importlib.metadata
import os
import subprocess
import time
from common import (
    DEFAULT_CONFIG,
    asset_record,
    as_config,
    emit,
    metrics,
    output_dir,
    positive,
    write_json,
)


def load_pipeline(cfg):
    import torch
    from diffusers import SanaPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a GPU runtime with CUDA-enabled PyTorch")
    pipe = SanaPipeline.from_pretrained(
        cfg["model"],
        revision=cfg.get("model_revision"),
        variant=cfg.get("variant", "fp16"),
        torch_dtype=torch.float16,
    ).to("cuda")
    # Sana recipe: fp16 transformer, bf16 text encoder and VAE to prevent black images.
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run(cfg=DEFAULT_CONFIG, n=1, out=None, mode="job"):
    import torch

    cfg = as_config(cfg)
    if cfg.get("model_revision") is None:
        raise ValueError("Pin model_revision to a Hub commit")
    cfg["image"] = os.environ.get("WORKER_IMAGE")
    folder = output_dir(out)
    m = metrics(cfg, mode)
    emit(folder, "run.started")
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        pipe = load_pipeline(cfg)
        torch.cuda.synchronize()
        m["load_s"] = round(time.perf_counter() - start, 4)
        m["env"].update(
            gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            torch=torch.__version__,
            diffusers=importlib.metadata.version("diffusers"),
            preset=os.environ.get("GPU_PRESET"),
        )
        try:
            m["env"]["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
        emit(folder, "model.loaded", duration_s=m["load_s"])
        for index in range(n):
            path = folder / f"asset-{index:03d}.png"
            torch.cuda.synchronize()
            start = time.perf_counter()
            image = pipe(
                prompt=cfg["prompt"],
                **cfg["params"],
                generator=torch.Generator(device="cuda").manual_seed(cfg["seed"]),
            ).images[0]
            torch.cuda.synchronize()
            m["latency_s"].append(round(time.perf_counter() - start, 4))
            peak_bytes = torch.cuda.max_memory_allocated()
            m["env"]["peak_allocated_bytes"] = peak_bytes
            m["env"]["peak_vram_gb"] = round(peak_bytes / 1e9, 3)
            image.save(path)
            m["assets"].append(asset_record(path))
            emit(folder, "asset.saved", file=path.name)
        m["status"] = "completed"
    except Exception as exc:
        m["status"] = "failed"
        emit(folder, "run.failed", error_type=type(exc).__name__)
        raise
    finally:
        emit(folder, "run.finished", status=m["status"])
        write_json(folder / "metrics.json", m)
        print(folder / "metrics.json")
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--n", type=positive, default=1)
    p.add_argument("--out")
    p.add_argument("--mode", choices=["job", "jupyterlab"], default="job")
    a = p.parse_args()
    run(a.config, a.n, a.out, a.mode)


if __name__ == "__main__":
    main()
