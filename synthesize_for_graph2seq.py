import os
import sys
import json
import argparse
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenAgentQG
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
sys.path.insert(0, PROJECT_ROOT)
from data_loader import load_mhqg_json, triples_to_text
from fusion import build_fusion_graph
from agentic.agents import batch_generate_questions_from_subgraph_only, set_template_protocol, set_current_dataset
from agentic.communication import TemplateLibraryProtocol


def _synthesize_one_chunk(chunk, per_call_batch_size):
    """单批：供线程池调用。"""
    if not chunk:
        return []
    return batch_generate_questions_from_subgraph_only(
        chunk, batch_size=min(per_call_batch_size, len(chunk))
    )


def synthesize_split(samples, dataset_name, split_name, max_samples=None, batch_size=10, parallel=8):
    """对一组样本仅用子图生成问句；返回 list of {inGraph, answers, answer_ids, outSeq, qId}（outSeq=合成问句）。

    batch_size: 每次 API 请求处理的样本数
    parallel: 并发请求数。若遇 API 限流(429) 可适当减小（如 4）。
    """
    from agentic.agents import normalize_question_for_eval

    protocol = TemplateLibraryProtocol(DATA_ROOT)
    protocol.load(dataset_name)
    set_template_protocol(protocol)
    set_current_dataset(dataset_name)

    if max_samples:
        samples = samples[:max_samples]
    fusions = []
    for s in samples:
        triples = s.get("triples", [])
        entities = s.get("entities", set())
        fusions.append(build_fusion_graph(triples, entities, "", []))

    # 按 batch_size 切成多个 chunk
    chunks = [fusions[i : i + batch_size] for i in range(0, len(fusions), batch_size)]
    questions = []

    if parallel and parallel > 1 and len(chunks) > 1:
        # 多线程并发调用 LLM，每个 chunk 一次请求
        results = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            future_to_idx = {
                ex.submit(_synthesize_one_chunk, ch, batch_size): idx
                for idx, ch in enumerate(chunks)
            }
            for fut in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc=f"{split_name} synthesize",
            ):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = [""] * len(chunks[idx])
        for idx in range(len(chunks)):
            questions.extend(results[idx] or [""] * len(chunks[idx]))
    else:
        # 顺序执行（调试或小规模）
        for ch in tqdm(chunks, desc=f"{split_name} synthesize"):
            qs = _synthesize_one_chunk(ch, batch_size)
            questions.extend(qs)

    # 组装为 Graph2Seq 格式：保留 inGraph, answers, answer_ids, qId；outSeq = 合成问句
    out = []
    for i, s in enumerate(samples):
        rec = {
            "inGraph": s["inGraph"],
            "answers": s.get("answers", []),
            "answer_ids": s.get("answer_ids", []),
            "qId": s.get("qId", i),
            "outSeq": normalize_question_for_eval(questions[i]) if i < len(questions) else "",
        }
        out.append(rec)
    return out


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Synthesize QA from subgraph-only (zero-shot), then optionally train Graph2Seq")
    parser.add_argument("--dataset", type=str, default="mhqg-wq",
                        choices=["mhqg-wq", "mhqg-pq", "mhqg-wq-inkg", "mhqg-pq-inkg", "mhqg-wq-text", "mhqg-pq-text"])
    parser.add_argument("--data_dir", type=str, default=None, help="Data root (default: OpenAgentQG/data)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir for synthetic JSONL (default: data/<dataset>/synthetic)")
    parser.add_argument("--max_per_split", type=int, default=None, help="Cap samples per split (for quick runs)")
    parser.add_argument("--batch_size", type=int, default=10, help="Samples per API call (default 10)")
    parser.add_argument("--parallel", type=int, default=8, help="Concurrent API calls (default 8)")
    parser.add_argument("--train_graph2seq", action="store_true", help="After synthesis, train Graph2Seq and run test (feedback)")
    args = parser.parse_args()

    data_dir = args.data_dir or DATA_ROOT
    out_dir = args.out_dir or os.path.join(data_dir, args.dataset, "synthetic")
    batch_size = args.batch_size

    for split in ["train", "dev", "test"]:
        path = os.path.join(data_dir, args.dataset, f"{split}.json")
        if not os.path.isfile(path):
            print(f"Skip {split}: no {path}")
            continue
        samples = load_mhqg_json(path, max_samples=args.max_per_split)
        if not samples:
            continue
        print(f"{split}: {len(samples)} samples")
        records = synthesize_split(samples, args.dataset, split, max_samples=args.max_per_split, batch_size=batch_size, parallel=args.parallel)
        out_path = os.path.join(out_dir, f"{split}.json")
        write_jsonl(records, out_path)
        print(f"  -> {out_path}")

    if not args.train_graph2seq:
        print("Done. Run with --train_graph2seq to train Graph2Seq and get feedback.")
        return

    # 调用 agentic.quality_assessment 中的 G2S 组件（论文 3.3.4 Quality Assessment）
    from agentic.quality_assessment.graph2seq_runner import run_graph2seq_quality_assessment
    from config import GRAPH2SEQ_ROOT
    result = run_graph2seq_quality_assessment(
        synthetic_data_dir=out_dir,
        dataset=args.dataset,
        graph2seq_root=GRAPH2SEQ_ROOT,
    )
    if result and result.get("metrics"):
        metrics = result["metrics"]
        print("\n--- Quality Assessment (Graph2Seq on synthetic data) ---")
        print("Feedback for next round:", metrics)
    else:
        print("Quality Assessment skipped or failed (check GRAPH2SEQ_ROOT and Graph2Seq4TKGQG).")


if __name__ == "__main__":
    main()
