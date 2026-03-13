"""
Run mhqg-wq (WQ) and mhqg-pq (PQ). BLEU_1..4, ROUGE_L.
Usage: python run_both.py [--split test] [--max_samples N] [--parallel P]
"""
import os
import sys
import argparse
import time
import concurrent.futures
from tqdm import tqdm

from data_loader import load_mhqg_json, load_gold_for_eval
from pipeline import run_open_agent_qg
from evaluation import compute_metrics, format_like_graph2seq
from agentic.agents import normalize_question_for_eval, set_template_protocol, set_current_dataset
from agentic.communication import TemplateLibraryProtocol, reset_session_pool
import tokens_cal
from config import DATA_ROOT, MAX_WORKERS


ALL_DATASETS = [
    ("mhqg-wq", "WQ"),
    ("mhqg-pq", "PQ"),
    ("mhqg-wq-inkg", "WQ-IncKG"),
    ("mhqg-pq-inkg", "PQ-IncKG"),
    ("mhqg-wq-text", "WQ-Text"),
    ("mhqg-pq-text", "PQ-Text"),
]
DATASETS = ALL_DATASETS[:2]  # default: WQ/PQ only


def run_dataset(data_dir, dataset, split, output_dir, max_samples, parallel):
    data_path = os.path.join(data_dir, dataset, f"{split}.json")
    if not os.path.isfile(data_path):
        print(f"[{dataset}] Skip: {data_path} not found")
        return None
    samples = load_mhqg_json(data_path, max_samples=max_samples)
    if not samples:
        return None
    reset_session_pool()
    protocol = TemplateLibraryProtocol(data_dir)
    protocol.load(dataset)
    set_template_protocol(protocol)
    set_current_dataset(dataset)
    gold = load_gold_for_eval(data_dir, dataset, split)
    if gold is None:
        gold = [s.get("outSeq") or "" for s in samples]
    if len(gold) != len(samples):
        gold = (gold + [""] * len(samples))[:len(samples)]
    predictions = []
    if parallel <= 0:
        for sample in tqdm(samples, desc=dataset, leave=False):
            try:
                out = run_open_agent_qg(sample)
                predictions.append(out["question"])
            except Exception as e:
                predictions.append("")
                tqdm.write(f"[{dataset}] error: {e}")
    else:
        batch_size = min(parallel, MAX_WORKERS)
        results = [None] * len(samples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(run_open_agent_qg, s): i for i, s in enumerate(samples)}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=dataset, leave=False):
                i = futures[fut]
                try:
                    out = fut.result()
                    results[i] = out["question"]
                except Exception as e:
                    results[i] = ""
                    tqdm.write(f"[{dataset}] sample {i}: {e}")
        predictions = results
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, f"pred_{dataset}_{split}.txt")
    gold_path = os.path.join(output_dir, f"gold_{dataset}_{split}.txt")
    predictions = [normalize_question_for_eval(p) for p in predictions]
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write((p or "").strip() + "\n")
    with open(gold_path, "w", encoding="utf-8") as f:
        for g in gold:
            f.write((g or "").strip() + "\n")
    metrics = compute_metrics(gold, predictions)
    return {
        "dataset": dataset,
        "split": split,
        "n": len(samples),
        "metrics": metrics,
        "pred_path": pred_path,
    }


def main():
    parser = argparse.ArgumentParser(description="OpenAgentQG: run WQ/PQ, evaluate BLEU/ROUGE")
    parser.add_argument("--data_dir", type=str, default=None, help="Data dir (default: OpenAgentQG/data)")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--parallel", type=int, default=0, help="Parallel workers (0=sequential)")
    parser.add_argument("--datasets", type=str, default="base", choices=["base", "all"],
                        help="base=WQ+PQ; all=6 datasets")
    args = parser.parse_args()
    data_dir = args.data_dir or DATA_ROOT
    if not os.path.isdir(data_dir):
        print(f"Data dir not found: {data_dir}")
        sys.exit(1)
    datasets_to_run = ALL_DATASETS if args.datasets == "all" else DATASETS
    total_tokens = 0
    start = time.time()
    results = []
    for dataset, label in datasets_to_run:
        print(f"\n========== {label} ({dataset}) ==========")
        res = run_dataset(
            data_dir=data_dir,
            dataset=dataset,
            split=args.split,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            parallel=args.parallel,
        )
        if res is None:
            continue
        results.append(res)
        # Output format aligned with Graph2Seq
        line = format_like_graph2seq(res["n"], res["metrics"])
        print(line)
        print("Saved predictions to", res["pred_path"])
    elapsed = time.time() - start
    total_tokens = tokens_cal.get_tokens()
    print("\n========== Summary ==========")
    for res in results:
        m = res["metrics"]
        print(f"{res['dataset']}: BLEU_4 = {m['Bleu_4']:.3f} | ROUGE_L = {m['ROUGE_L']:.3f} | Overall = {m['Overall']:.3f}")
    print(f"Total tokens: {total_tokens} | Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
