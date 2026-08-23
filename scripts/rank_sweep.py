"""B4 bonus — controlled rank sweep at a FIXED placement (text-linear).

r=16 is already `correct` from NB3 (target score already in results/verdict.json).
This trains r=8 and r=64 with everything else identical to `correct` -- same
placement, same LR, same step budget, same mask -- then scores all three on the
target set the same way NB5 does, and writes results/rank_sweep.json.

Run from the repo root after NB1-NB3 have produced data/split/ and adapters/correct/:
    python scripts/rank_sweep.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from labkit import data, evaluate as ev, generate, modeling, report, train
from labkit.config import SPECS, get_tier, training_epochs

TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
RANKS = [8, 64]          # 16 == `correct`, already trained


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def train_rank(r: int) -> dict:
    key = f"rank_sweep_r{r}"
    out = ROOT / "adapters" / key
    if (out / "adapter_model.safetensors").exists():
        print(f"skip {key}: already trained")
        return {"run": key, "r": r}

    base = SPECS["correct"]
    spec = type(base)(key=key, r=r, alpha=2 * r, target="text-linear", lr=base.lr,
                      load_in_4bit=False, label=f"text-linear · r={r} · LR 10x · 16-bit",
                      teaches=f"B4 bonus: controlled rank sweep, r={r} at fixed placement.")

    model, tok = generate.load_base(TIER)
    train_rows = load_jsonl(ROOT / "data" / "split" / "train.jsonl")
    train_ds = Dataset.from_list(
        data.to_training_dataset(tok, train_rows, max_length=TIER.max_length,
                                 mask_mode=os.environ.get("MASK_MODE", "assistant-only")))
    targets = modeling.resolve_target_modules(model, spec.target)
    trainable = modeling.count_lora_params(model, targets, spec.r)
    max_steps = train.planned_steps(len(train_ds), TIER, training_epochs())

    want = train.sft_config_kwargs(TIER, spec, str(out), max_steps=max_steps)
    sft_kwargs, _ = train.filter_kwargs(SFTConfig, want, label=f"SFTConfig[{key}]")
    lora_kwargs, _ = train.filter_kwargs(
        LoraConfig, train.lora_config_kwargs(spec, targets), label=f"LoraConfig[{key}]")

    trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs),
                         train_dataset=train_ds, processing_class=tok,
                         peft_config=LoraConfig(**lora_kwargs))
    fix = train.align_trainable_precision(trainer.model)
    if fix.get("recast"):
        print(f"  precision fix: recast {fix['recast']}/{fix['trainable_tensors']}")

    t0 = time.perf_counter()
    res = trainer.train()
    elapsed = time.perf_counter() - t0
    trainer.model.save_pretrained(out)

    row = train.summarize_run(spec, TIER, targets, trainable, elapsed, generate.peak_vram_gb())
    row["final_loss"] = round(res.training_loss, 4)
    row["max_steps"] = max_steps
    row["teaches"] = spec.teaches
    report.append_row(row, results_dir=ROOT / "results")

    del trainer, model
    generate.free_memory()
    return row


def score_rank(key: str, r: int, target: list[dict]) -> dict:
    adapter_dir = ROOT / "adapters" / key
    model, tok = generate.load_base(TIER)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    preds, lat = generate.generate_batch(
        model, tok, [row["input"] for row in target], system=generate.NAIVE_PROMPT,
        label=f"{key}/target")
    tgt = sum(ev.triage_field_accuracy(p, row["label"]) for p, row in zip(preds, target)) / len(target)
    fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)
    del model
    generate.free_memory()
    return {"run": key, "r": r, "target": round(tgt, 4), "format": round(fmt, 4),
            "latency_ms": round(lat, 1), "n": len(target)}


def main():
    for r in RANKS:
        print("=" * 70)
        print(f"RANK SWEEP r={r} (text-linear, matches `correct` in every other way)")
        train_rank(r)

    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    correct_verdict = json.loads((ROOT / "results" / "verdict.json").read_text(encoding="utf-8"))
    correct_row = next(c for c in correct_verdict["comparison"] if c["run"] == "(c) LoRA fine-tune")

    rows = [{"run": "correct", "r": 16, "target": correct_row["target"],
             "format": correct_row["format"], "latency_ms": correct_row["latency_ms"],
             "n": correct_row["n"]}]
    for r in RANKS:
        rows.append(score_rank(f"rank_sweep_r{r}", r, target))
    rows.sort(key=lambda row: row["r"])

    print()
    print(report.markdown_table(rows, ["run", "r", "target", "format", "latency_ms", "n"]))
    report.write_json(rows, "rank_sweep.json", results_dir=ROOT / "results")


if __name__ == "__main__":
    main()
