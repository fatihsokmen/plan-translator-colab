"""Fine-tunes a small instruct model to turn a question into a plan in the app's contract (ModelPlan v2).

Each training row is {"system", "input", "output"}; the model learns to write `output` and the loss is
computed on it only. Inputs are built by the app's own ModelPlan.prompt, so training and the app match.

    pip install -r training/requirements.txt
    python training/train.py --model google/gemma-3-270m-it --data training/data --out runs/gemma270m

Defaults follow the nl2jq recipe (LoRA r=32, alpha=64, all linear layers, 1 epoch, cosine, lr 1e-4).
Choose checkpoints on the dev split only; the frozen test set is scored once, by the app (see README).
"""
import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer


def to_prompt_completion(row):
    return {
        "prompt": [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["input"]}],
        "completion": [{"role": "assistant", "content": row["output"]}],
    }


def dev_exact_match(model, tok, rows, limit=300):
    """Share of dev rows whose generated plan equals the label as JSON, and share that parse at all."""
    model.eval()
    exact = valid = 0
    for row in rows[:limit]:
        msgs = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["input"]}]
        # enable_thinking=False keeps Qwen3 from writing a reasoning block; other templates ignore it.
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=160, do_sample=False)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            got = json.loads(text[text.index("{"): text.rindex("}") + 1])
            valid += 1
            exact += got == json.loads(row["output"])
        except ValueError:
            pass
    n = min(limit, len(rows))
    return exact / n, valid / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--data", default="training/data")
    ap.add_argument("--out", default="runs/translator")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="use only this many training rows (smoke tests)")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--dev_rows", type=int, default=300, help="dev rows generated for the exact-match check")
    ap.add_argument("--eval_rows", type=int, default=200, help="dev rows used for the loss during training")
    ap.add_argument("--save_steps", type=int, default=500)
    args = ap.parse_args()

    files = {"train": f"{args.data}/train/train.jsonl", "dev": f"{args.data}/dev/dev.jsonl"}
    raw = load_dataset("json", data_files=files)
    if args.limit:
        raw["train"] = raw["train"].shuffle(seed=args.seed).select(range(args.limit))
    ds = raw.map(to_prompt_completion, remove_columns=raw["train"].column_names)

    tok = AutoTokenizer.from_pretrained(args.model)
    # is_bf16_supported() also says yes for emulated bf16 (T4), which the trainer then rejects; ask for an Ampere or newer GPU.
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if bf16 else torch.float32)

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=args.save_steps,
        save_total_limit=2,
        max_length=args.max_len,
        completion_only_loss=True,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        report_to="none",
        seed=args.seed,
    )
    peft = None if args.full else LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear", task_type="CAUSAL_LM")
    eval_ds = ds["dev"].shuffle(seed=args.seed).select(range(min(args.eval_rows, len(ds["dev"]))))
    trainer = SFTTrainer(model=model, args=config, train_dataset=ds["train"], eval_dataset=eval_ds, processing_class=tok, peft_config=peft)
    # Rerunning with the same --out continues from its last checkpoint, so a lost session costs at most save_steps.
    last = get_last_checkpoint(args.out) if os.path.isdir(args.out) else None
    if last:
        print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)

    final = trainer.model.merge_and_unload() if peft else trainer.model
    final.save_pretrained(f"{args.out}/final")
    tok.save_pretrained(f"{args.out}/final")

    dev_rows = list(raw["dev"])
    random.Random(args.seed).shuffle(dev_rows)
    exact, valid = dev_exact_match(final, tok, dev_rows, limit=args.dev_rows)
    print(f"dev: exact plan {exact:.3f} · valid JSON {valid:.3f} ({min(args.dev_rows, len(dev_rows))} random dev rows)")


if __name__ == "__main__":
    main()
