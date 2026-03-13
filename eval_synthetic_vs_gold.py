#!/usr/bin/env python
import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from evaluation import compute_metrics
from agentic.agents import normalize_question_for_eval


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", required=True, help="合成数据 JSONL（含 outSeq, qId）")
    p.add_argument("--gold", required=True, help="标准数据 JSONL（含 outSeq, qId）")
    p.add_argument("--split", default="test", help="仅用于打印")
    args = p.parse_args()

    gold_list = load_jsonl(args.gold)
    gold_by_id = {r["qId"]: r.get("outSeq", "").strip() for r in gold_list}

    syn_list = load_jsonl(args.synthetic)
    gold_refs = []
    preds = []
    for r in syn_list:
        qid = r.get("qId")
        if qid not in gold_by_id:
            continue
        gold_refs.append(gold_by_id[qid])
        preds.append(normalize_question_for_eval(r.get("outSeq", "") or ""))

    if not gold_refs:
        print("No overlapping qId between synthetic and gold. Exit.")
        return
    print(f"[{args.split}] 对齐样本数: {len(gold_refs)} (synthetic 共 {len(syn_list)} 条)")
    metrics = compute_metrics(gold_refs, preds)
    print("Bleu_1 = {:.3f} | Bleu_2 = {:.3f} | Bleu_3 = {:.3f} | Bleu_4 = {:.3f} | ROUGE_L = {:.3f} | Overall = {:.3f}".format(
        metrics.get("Bleu_1", 0), metrics.get("Bleu_2", 0), metrics.get("Bleu_3", 0),
        metrics.get("Bleu_4", 0), metrics.get("ROUGE_L", 0), metrics.get("Overall", 0),
    ))


if __name__ == "__main__":
    main()
