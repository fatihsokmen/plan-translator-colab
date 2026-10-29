"""Writes a model's plans for the frozen test prompts; the app scores them.

    python training/predict.py --model runs/gemma270m/final \
        --prompts training/data/test/real_like_v2.prompts.jsonl \
        --out training/data/predictions/gemma270m-v1.jsonl

Then on the computer with the repo: ./gradlew :shared:jvmTest --tests "*RealLikeTestSetTest*" -i
prints correct / no figure / WRONG for every file in training/data/predictions/. Score each model once.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    with open(args.prompts) as f, open(args.out, "w") as out:
        for line in f:
            row = json.loads(line)
            msgs = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["input"]}]
            # enable_thinking=False keeps Qwen3 from writing a reasoning block; other templates ignore it.
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({"id": row["id"], "output": text}) + "\n")


if __name__ == "__main__":
    main()
