"""
OpenAgentQG full pipeline: Fusion -> Agentic synthesis -> G2S train -> quality select.
State in run_dir: pipeline.log, run_state.json. Use --resume to continue.
Usage: python run_full_pipeline.py --dataset mhqg-wq [--resume --run_id iter_1]
"""
import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
sys.path.insert(0, PROJECT_ROOT)

from data_loader import load_mhqg_json, inGraph_to_triples, triples_to_text
from agentic.agents import set_template_protocol, set_current_dataset, set_extra_example_bank
from agentic.communication import TemplateLibraryProtocol
from evaluation import compute_metrics

RUN_STATE_FILE = "run_state.json"
PIPELINE_LOG_FILE = "pipeline.log"


def _log(run_dir, msg, also_stdout=True):
    """Write to run_dir/pipeline.log, optionally stdout."""
    if also_stdout:
        print(msg)
    if run_dir:
        log_path = os.path.join(run_dir, PIPELINE_LOG_FILE)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(datetime.now().isoformat() + " " + msg + "\n")
        except Exception:
            pass


def _load_run_state(run_dir):
    path = os.path.join(run_dir, RUN_STATE_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_run_state(run_dir, state):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, RUN_STATE_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _write_jsonl(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stage1_fusion_and_agentic(data_dir, dataset, split, max_samples, batch_size, parallel, out_synthetic_dir):
    """Fusion + Agentic: build_fusion_graph, synthesize questions per subgraph."""
    from synthesize_for_graph2seq import synthesize_split
    path = os.path.join(data_dir, dataset, f"{split}.json")
    if not os.path.isfile(path):
        return []
    samples = load_mhqg_json(path, max_samples=max_samples)
    if not samples:
        return []
    records = synthesize_split(
        samples, dataset, split,
        max_samples=max_samples,
        batch_size=batch_size,
        parallel=parallel,
    )
    out_path = os.path.join(out_synthetic_dir, f"{split}.json")
    _write_jsonl(records, out_path)
    return records


def run_agentic_synthesis(data_dir, dataset, output_root, run_id, max_per_split, batch_size, parallel, extra_example_bank=None):
    """Agentic 阶段：各 split 基于子图合成问题，结果与过程通过通讯协议存到指定文件夹。"""
    protocol = TemplateLibraryProtocol(data_dir)
    protocol.load(dataset)
    set_template_protocol(protocol)
    set_current_dataset(dataset)

    run_dir = os.path.join(output_root, run_id)
    synthetic_dir = os.path.join(run_dir, "synthetic")
    os.makedirs(synthetic_dir, exist_ok=True)

    # Inject G2S refined into example_bank for iteration
    if extra_example_bank:
        set_extra_example_bank(extra_example_bank)
    else:
        set_extra_example_bank([])

    for split in ["train", "dev", "test"]:
        recs = stage1_fusion_and_agentic(
            data_dir, dataset, split,
            max_samples=max_per_split,
            batch_size=batch_size,
            parallel=parallel,
            out_synthetic_dir=synthetic_dir,
        )
        if recs:
            print(f"  [{split}] wrote {len(recs)} synthetic to {synthetic_dir}/{split}.json")

    protocol.save_to_folder(run_dir)
    print(f"  Protocol saved to {run_dir}/protocol_dump.json")
    return run_dir, synthetic_dir


def _merge_test_with_gold(test_json_path, gold_txt_path, data_dir, dataset):
    """合并 test.json 与 eval_gold/test.txt，生成含 outSeq 的临时文件供 G2S 使用。"""
    import tempfile
    gold_lines = []
    with open(gold_txt_path, "r", encoding="utf-8") as f:
        gold_lines = [line.strip() for line in f]
    records = []
    with open(test_json_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["outSeq"] = gold_lines[i] if i < len(gold_lines) else ""
            records.append(rec)
    out_dir = os.path.join(data_dir, dataset, "agentic_output", "_g2s_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_with_gold.json")
    _write_jsonl(records, out_path)
    return out_path


def run_g2s_train_and_test(synthetic_dir, data_dir, dataset, train_val_ratio, graph2seq_root):
    """G2S train on synthetic, test on full QA; return metrics and predictions path."""
    from agentic.quality_assessment.graph2seq_runner import run_graph2seq_quality_assessment
    from config import GRAPH2SEQ_ROOT
    root = graph2seq_root or GRAPH2SEQ_ROOT
    # 优先使用含 outSeq 的 test_with_gold.json；若无则从 test.json + eval_gold 合并
    original_test_path = os.path.join(data_dir, dataset, "test_with_gold.json")
    if not os.path.isfile(original_test_path):
        test_path = os.path.join(data_dir, dataset, "test.json")
        gold_path = os.path.join(data_dir, dataset, "eval_gold", "test.txt")
        if os.path.isfile(test_path) and os.path.isfile(gold_path):
            original_test_path = _merge_test_with_gold(test_path, gold_path, data_dir, dataset)
        else:
            original_test_path = test_path
    result = run_graph2seq_quality_assessment(
        synthetic_data_dir=synthetic_dir,
        dataset=dataset,
        graph2seq_root=root,
        train_val_ratio=train_val_ratio,
        return_predictions_path=True,
        original_test_path=original_test_path if os.path.isfile(original_test_path) else None,
    )
    return result


def refined_to_example_bank(refined_records):
    """Convert refined records to example_bank format for set_extra_example_bank."""
    out = []
    for rec in refined_records or []:
        triples_list, _ = inGraph_to_triples(rec.get("inGraph", {}))
        triples_str = triples_to_text(triples_list)
        ans = rec.get("answers", [])
        ans_str = ", ".join(ans) if isinstance(ans, (list, tuple)) else str(ans)
        q = (rec.get("outSeq") or "").strip()
        if triples_str and q:
            out.append({"triples": triples_str, "answers": ans_str, "question": q})
    return out


def merge_refined_into_synthetic_train(synthetic_dir, refined_records):
    """Merge refined into synthetic train.json for next G2S train."""
    train_path = os.path.join(synthetic_dir, "train.json")
    existing = []
    if os.path.isfile(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
    merged = existing + list(refined_records)
    _write_jsonl(merged, train_path)
    print(f"  Merged {len(refined_records)} refined into train ({len(merged)} total)")


def quality_select_and_save_refined(test_records, predictions_path, run_dir, quality_metric, quality_threshold):
    """Filter by quality, save refined for next iteration."""
    if not test_records or not predictions_path or not os.path.isfile(predictions_path):
        return []
    with open(predictions_path, "r", encoding="utf-8") as f:
        pred_lines = [line.strip() for line in f if line.strip()]

    metric_key = "Bleu_4" if quality_metric.upper().startswith("B") else "ROUGE_L"
    if metric_key not in ("Bleu_4", "ROUGE_L"):
        metric_key = "Bleu_4"

    refined = []
    for i, rec in enumerate(test_records):
        if i >= len(pred_lines):
            break
        gold = (rec.get("outSeq") or "").strip()
        pred = (pred_lines[i] or "").strip()
        if not pred:
            continue
        scores = compute_metrics([gold], [pred])
        score = scores.get(metric_key, 0.0)
        if score >= quality_threshold:
            refined.append({
                "inGraph": rec.get("inGraph"),
                "answers": rec.get("answers", []),
                "answer_ids": rec.get("answer_ids", []),
                "qId": rec.get("qId", i),
                "outSeq": pred,
            })
    out_path = os.path.join(run_dir, "refined_train.json")
    _write_jsonl(refined, out_path)
    print(f"  Refined: {len(refined)} / {len(test_records)} above {metric_key}>={quality_threshold}, saved to {out_path}")
    return refined


def main():
    parser = argparse.ArgumentParser(description="OpenAgentQG full pipeline: Fusion -> Agentic -> G2S -> Select")
    parser.add_argument("--dataset", type=str, default="mhqg-wq",
                        help="单个数据集，或 all=依次跑全部6个")
    parser.add_argument("--data_dir", type=str, default=None, help="Data root (default: OpenAgentQG/data)")
    parser.add_argument("--output_root", type=str, default=None, help="Output root (default: data/<dataset>/agentic_output)")
    parser.add_argument("--run_id", type=str, default=None, help="Run id for resume")
    parser.add_argument("--max_per_split", type=int, default=None, help="Max samples per split")
    parser.add_argument("--train_val_ratio", type=float, default=0.8, help="Train ratio for G2S (val = 1-ratio from train)")
    parser.add_argument("--quality_metric", type=str, default="Bleu_4", choices=["Bleu_4", "ROUGE_L"])
    parser.add_argument("--quality_threshold", type=float, default=15.0, help="Keep samples with metric >= this")
    parser.add_argument("--iterations", type=int, default=2, help="Iteration count")
    parser.add_argument("--batch_size", type=int, default=20, help="Samples per API call")
    parser.add_argument("--parallel", type=int, default=8, help="Parallel API workers")
    parser.add_argument("--graph2seq_root", type=str, default=None)
    parser.add_argument("--load_prev_run", type=str, default=None, help="Load protocol from previous run_dir for iteration")
    parser.add_argument("--skip_g2s", action="store_true", help="Skip G2S train/test; only run synthesis + protocol")
    parser.add_argument("--resume", action="store_true", help="Resume from run_state.json")
    args = parser.parse_args()

    ALL_DATASETS = ["mhqg-wq", "mhqg-pq", "mhqg-wq-inkg", "mhqg-pq-inkg", "mhqg-wq-text", "mhqg-pq-text"]
    datasets_to_run = ALL_DATASETS if args.dataset.lower() == "all" else [args.dataset]
    for ds in datasets_to_run:
        if ds not in ALL_DATASETS:
            print(f"Unknown dataset: {ds}, skip")
            continue
        _run_one_dataset(args, ds)

    if len(datasets_to_run) > 1:
        print("\n--- All datasets done ---")


def _run_one_dataset(args, dataset):
    """对单个数据集跑完整 pipeline（含迭代）。"""
    args = argparse.Namespace(**{k: getattr(args, k) for k in vars(args)})
    args.dataset = dataset

    data_dir = args.data_dir or DATA_ROOT
    output_root = args.output_root or os.path.join(data_dir, args.dataset, "agentic_output")
    num_iters = max(1, args.iterations)
    prev_refined = []  # G2S refined from previous iteration

    for iter_num in range(1, num_iters + 1):
        run_id = f"iter_{iter_num}" if num_iters > 1 else (args.run_id or datetime.now().strftime("iter_%Y%m%d_%H%M%S"))
        run_dir = os.path.join(output_root, run_id)
        os.makedirs(run_dir, exist_ok=True)
        if iter_num == 1:
            _tee_stdout_to_log(run_dir)
        state = _load_run_state(run_dir) if (args.resume and iter_num == 1) else {}

        def log(msg):
            _log(run_dir, msg)

        log(f"\n{'='*50} Iteration {iter_num}/{num_iters} {'='*50}")
        log(f"[start] dataset={args.dataset} run_id={run_id} max_per_split={args.max_per_split}")

        if args.load_prev_run and iter_num == 1 and os.path.isdir(args.load_prev_run):
            protocol = TemplateLibraryProtocol(data_dir)
            protocol.load(args.dataset)
            protocol.load_runtime_from_folder(args.load_prev_run, merge=True)
            set_template_protocol(protocol)
            set_current_dataset(args.dataset)
            log(f"Loaded protocol from {args.load_prev_run} for this run.")
        elif iter_num == 1:
            protocol = TemplateLibraryProtocol(data_dir)
            protocol.load(args.dataset)
            set_template_protocol(protocol)
            set_current_dataset(args.dataset)

        # 迭代时注入上一轮 G2S refined 到 example_bank
        extra_example_bank = refined_to_example_bank(prev_refined) if prev_refined else None

        # Stage 1 & 2: Fusion + Agentic
        synthetic_dir = os.path.join(run_dir, "synthetic")
        if state.get("stage1_2_done") and os.path.isdir(synthetic_dir) and os.path.isfile(os.path.join(synthetic_dir, "train.json")):
            log("--- Stage 1–2 skipped (resume: already done), synthetic_dir=" + synthetic_dir)
        else:
            log("\n--- Stage 1–2: Fusion + Agentic (subgraph -> synthetic QA) ---")
            run_dir, synthetic_dir = run_agentic_synthesis(
                data_dir, args.dataset, output_root, run_id,
                max_per_split=args.max_per_split,
                batch_size=args.batch_size,
                parallel=args.parallel,
                extra_example_bank=extra_example_bank,
            )
            state["stage1_2_done"] = True
            state["synthetic_dir"] = synthetic_dir
            state["run_dir"] = run_dir
            state["stage1_2_finished_at"] = datetime.now().isoformat()
            _save_run_state(run_dir, state)
            log("Stage 1–2 done. State saved.")

        # Merge refined into synthetic train for G2S
        if iter_num > 1 and prev_refined:
            merge_refined_into_synthetic_train(synthetic_dir, prev_refined)

        # Stage 3: G2S
        result = None
        if args.skip_g2s:
            log("\n--- Stage 3: G2S skipped (--skip_g2s). ---")
            break
        if state.get("stage3_done") and state.get("predictions_path") and os.path.isfile(state.get("predictions_path", "")):
            log("--- Stage 3 skipped (resume: already done), using saved predictions_path")
            result = {"metrics": state.get("g2s_metrics", {}), "predictions_path": state["predictions_path"]}
        else:
            log("\n--- Stage 3: G2S train on synthetic, test on standard QA full set ---")
            result = run_g2s_train_and_test(
                synthetic_dir, data_dir, args.dataset,
                train_val_ratio=args.train_val_ratio,
                graph2seq_root=args.graph2seq_root,
            )
            if result and result.get("metrics"):
                state["stage3_done"] = True
                state["g2s_metrics"] = result["metrics"]
                state["predictions_path"] = result.get("predictions_path") or ""
                state["stage3_finished_at"] = datetime.now().isoformat()
                _save_run_state(run_dir, state)
                log("Stage 3 done. State saved.")

        if not result or not result.get("metrics"):
            log("G2S stage failed or skipped. No refinement for this iteration.")
            break

        metrics = result["metrics"]
        log("G2S test metrics (on standard QA full set): " + json.dumps(metrics, ensure_ascii=False))

        # Stage 4: quality filter for next iteration
        test_path = os.path.join(data_dir, args.dataset, "test.json")
        test_records = []
        if os.path.isfile(test_path):
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        test_records.append(json.loads(line))
        prev_refined = quality_select_and_save_refined(
            test_records,
            result.get("predictions_path"),
            run_dir,
            args.quality_metric,
            args.quality_threshold,
        )
        state["stage4_done"] = True
        state["refined_count"] = len(prev_refined)
        state["stage4_finished_at"] = datetime.now().isoformat()
        _save_run_state(run_dir, state)

        log(f"Refined: {len(prev_refined)} samples for next iteration")
        log(f"Run dir: {run_dir} | Refined: {run_dir}/refined_train.json")

    log("\n--- Done ---")
    log("Resume: python run_full_pipeline.py --dataset " + args.dataset + " --resume --run_id iter_1")
    return


def _tee_stdout_to_log(run_dir):
    """Tee stdout to run_dir/stdout.log."""
    if not run_dir:
        return
    log_path = os.path.join(run_dir, "stdout.log")
    try:
        sys.stdout = _TeeWriter(sys.stdout, open(log_path, "a", encoding="utf-8"))
    except Exception:
        pass


class _TeeWriter:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()


if __name__ == "__main__":
    main()
