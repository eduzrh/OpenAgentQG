"""
OpenAgentQG: run and evaluate on mhqg-wq/mhqg-pq (BLEU-4, ROUGE-L).
Usage: python run.py --dataset mhqg-wq [--split dev] [--max_samples N] [--parallel 5]
"""
import os
import sys
import argparse
import time
import concurrent.futures
from tqdm import tqdm

from data_loader import load_mhqg_json, load_gold_for_eval
from pipeline import run_open_agent_qg, run_open_agent_qg_batch
from evaluation import compute_metrics, format_like_graph2seq
from agentic.agents import normalize_question_for_eval, set_template_protocol, set_current_dataset
import tokens_cal
from config import DATA_ROOT, MAX_WORKERS, BATCH_QS_PER_CALL
from agentic.communication import TemplateLibraryProtocol, reset_session_pool


def main():
    parser = argparse.ArgumentParser(description="OpenAgentQG: run and evaluate")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data (default: Graph2Seq4TKGQG/data)")
    parser.add_argument("--dataset", type=str, default="mhqg-wq",
                        choices=["mhqg-wq", "mhqg-pq", "mhqg-wq-inkg", "mhqg-pq-inkg", "mhqg-wq-text", "mhqg-pq-text"])
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap number of samples (default: all)")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--parallel", type=int, default=10, help="Parallel workers (0=sequential)")
    parser.add_argument("--batch_size", type=int, default=None, help="Questions per API call (default: config.BATCH_QS_PER_CALL)")
    args = parser.parse_args()

    # Data: OpenAgentQG/data (mhqg-wq=WQ, mhqg-pq=PQ)
    data_dir = args.data_dir or DATA_ROOT
    data_path = os.path.join(data_dir, args.dataset, f"{args.split}.json")
    if not os.path.isfile(data_path):
        print(f"Data not found: {data_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, f"pred_{args.dataset}_{args.split}.txt")
    gold_path = os.path.join(args.output_dir, f"gold_{args.dataset}_{args.split}.txt")

    print("Loading data...")
    samples = load_mhqg_json(data_path, max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples.")

    # Template protocol: load dataset template library (updated at runtime)
    protocol = TemplateLibraryProtocol(data_dir)
    protocol.load(args.dataset)
    set_template_protocol(protocol)
    set_current_dataset(args.dataset)

    batch_q = args.batch_size if args.batch_size is not None else BATCH_QS_PER_CALL
    batch_q = max(1, batch_q)

    tokens_cal.reset_tokens()
    reset_session_pool()
    start = time.time()
    predictions = []
    if args.parallel <= 0:
        if batch_q > 1:
            for start_i in range(0, len(samples), batch_q):
                chunk = samples[start_i:start_i + batch_q]
                out_list = run_open_agent_qg_batch(chunk, verbose=True)
                predictions.extend([r["question"] for r in out_list])
        else:
            for i, sample in enumerate(tqdm(samples, desc="OpenAgentQG")):
                if args.max_samples and len(samples) <= 20:
                    print(f"\n--- Sample {i+1}/{len(samples)} ---")
                out = run_open_agent_qg(sample, verbose=(args.max_samples is not None and len(samples) <= 20))
                predictions.append(out["question"])
    else:
        n_workers = min(args.parallel, MAX_WORKERS)
        print(f"Using {n_workers} parallel workers, batch {batch_q} questions per API call.")
        chunks = [samples[i:i + batch_q] for i in range(0, len(samples), batch_q)]
        results = [None] * len(chunks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_open_agent_qg_batch, ch): ci for ci, ch in enumerate(chunks)}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="OpenAgentQG"):
                ci = futures[fut]
                try:
                    out_list = fut.result()
                    results[ci] = [r["question"] for r in out_list]
                except Exception as e:
                    results[ci] = [""] * len(chunks[ci])
                    tqdm.write(f"Chunk {ci} error: {e}")
        for ci in range(len(chunks)):
            if results[ci] is not None:
                predictions.extend(results[ci])
        predictions = (predictions + [""] * len(samples))[:len(samples)]

    elapsed = time.time() - start
    total_tokens = tokens_cal.get_tokens()

    # Gold: from eval_gold/<split>.txt for evaluation
    gold = load_gold_for_eval(data_dir, args.dataset, args.split)
    if gold is None:
        gold = [s.get("outSeq") or "" for s in samples]
    if len(gold) != len(predictions):
        gold = (gold + [""] * len(predictions))[:len(predictions)]

    predictions = [normalize_question_for_eval(p) for p in predictions]
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write((p or "").strip() + "\n")
    with open(gold_path, "w", encoding="utf-8") as f:
        for g in gold:
            f.write((g or "").strip() + "\n")

    metrics = compute_metrics(gold, predictions)
    # Output format aligned with Graph2Seq
    print(format_like_graph2seq(len(samples), metrics))
    print("\n--- Summary ---")
    print(f"BLEU-4: {metrics.get('Bleu_4', 0):.2f} | ROUGE-L: {metrics.get('ROUGE_L', 0):.2f} | Overall: {metrics.get('Overall', 0):.2f}")
    print(f"\n--- Efficiency ---")
    print(f"Total tokens: {total_tokens}")
    print(f"Avg tokens/sample: {total_tokens / len(samples):.0f}" if samples else "N/A")
    print(f"Time: {elapsed:.1f}s")
    print(f"Avg time/sample: {elapsed / len(samples):.1f}s" if samples else "N/A")
    print("Saved predictions to", pred_path)


if __name__ == "__main__":
    main()
