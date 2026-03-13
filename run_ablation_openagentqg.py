"""
Run OpenAgentQG ablation experiments (Table 5 style, RoleAgentQG TKDE).

Paper mapping (Section 3.2 / 3.3):
- full                     : Full pipeline (Stage 1 neuro-symbolic fusion + Stage 2 agentic).
- no_fusion                : w/o Stage 1 — no Neuro-symbolic Unified Knowledge Fusion (triples only).
- no_agentic                : w/o Stage 2 — no Adaptive Agentic Collaborative Generation (one-shot Q only).
- no_meta_knowledge         : w/o 3.2.1 — no meta-symbolic aggregation, no meta-neural virtual nodes.
- no_graph_construction     : w/o 3.2.2 — no L_sym / H_neuro_sym layers (meta still computed, not used in graph).
- no_core_role_mgmt         : w/o 3.3.1 — fixed requirements, no Managing Editor decision.
- no_collaborative_decision : w/o 3.3.2 — max_iterations=1, no callback/revise loop.
- no_agentic_execution      : w/o 3.3.3 — no Contributor/Content/Copy chain, one-shot only.
- no_quality_assessment     : w/o 3.3.4 — no Editor-in-Chief score, return first edited question.

Usage:
  OPENAGENTQG_PAPER_FULL=1 python run_ablation_openagentqg.py [--split dev] [--max_samples N]
  python run_ablation_openagentqg.py --modes no_fusion no_agentic  # only specific modes
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# run from OpenAgentQG dir
_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluation import compute_metrics

ABLATION_MODES = [
    "full",           # OPENAGENTQG_ABLATION="" (full pipeline)
    "no_fusion",
    "no_agentic",
    "no_meta_knowledge",
    "no_graph_construction",
    "no_core_role_mgmt",
    "no_collaborative_decision",
    "no_agentic_execution",
    "no_quality_assessment",
]
DATASETS = ["mhqg-wq", "mhqg-pq"]


def run_one(dataset, split, output_dir, ablation_mode, max_samples=None):
    """Run run.py for one (dataset, split) with given ablation; env OPENAGENTQG_ABLATION set."""
    env = os.environ.copy()
    env["OPENAGENTQG_PAPER_FULL"] = "1"
    env["OPENAGENTQG_ABLATION"] = "" if ablation_mode == "full" else ablation_mode
    cmd = [
        sys.executable, "run.py",
        "--dataset", dataset,
        "--split", split,
        "--output_dir", output_dir,
    ]
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])
    try:
        subprocess.run(cmd, env=env, cwd=_SCRIPT_DIR, check=True, timeout=3600)
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] run.py failed: {e}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] run.py timeout", file=sys.stderr)


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def metrics_for_run(output_dir, dataset, split):
    """Load pred/gold from output_dir and return compute_metrics dict."""
    pred_path = os.path.join(output_dir, f"pred_{dataset}_{split}.txt")
    gold_path = os.path.join(output_dir, f"gold_{dataset}_{split}.txt")
    if not os.path.isfile(pred_path) or not os.path.isfile(gold_path):
        return None
    gold = read_lines(gold_path)
    pred = read_lines(pred_path)
    if len(pred) != len(gold):
        return None
    return compute_metrics(gold, pred)


def main():
    parser = argparse.ArgumentParser(description="OpenAgentQG ablation runner (Table 5 style)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap samples per run (e.g. 5 for quick test)")
    parser.add_argument("--output_base", type=str, default="ablation")
    parser.add_argument("--modes", nargs="+", default=None, help="Ablation modes to run (default: all)")
    parser.add_argument("--collect_only", action="store_true", help="Only collect metrics from existing output_dir")
    args = parser.parse_args()

    modes = args.modes if args.modes else ABLATION_MODES
    results = {}  # (mode, dataset) -> {"Bleu_4", "ROUGE_L", "Overall"}

    for mode in modes:
        if mode not in ABLATION_MODES:
            print(f"Unknown mode: {mode}", file=sys.stderr)
            continue
        output_dir = os.path.join(args.output_base, mode)
        if not args.collect_only:
            print(f"\n--- Ablation: {mode} ---")
            for ds in DATASETS:
                print(f"  Running {ds} {args.split} -> {output_dir}")
                run_one(ds, args.split, output_dir, mode, args.max_samples)

        for ds in DATASETS:
            m = metrics_for_run(output_dir, ds, args.split)
            if m is not None:
                results[(mode, ds)] = m
            else:
                results[(mode, ds)] = {"Bleu_4": 0.0, "ROUGE_L": 0.0, "Overall": 0.0}

    # Table 5 style: rows = modes, columns = WQ / PQ with BLEU-4, ROUGE-L, Overall
    print("\n" + "=" * 80)
    print("Ablation summary (Table 5 style) | BLEU-4 / ROUGE-L / Overall (0-100)")
    print("=" * 80)
    print(f"{'Mode':<28} | {'WebQ (WQ)':^24} | {'PathQ (PQ)':^24}")
    print(f"{'':28} | {'B-4':>6} {'R-L':>6} {'Ov':>6} | {'B-4':>6} {'R-L':>6} {'Ov':>6}")
    print("-" * 80)
    for mode in modes:
        rwq = results.get((mode, "mhqg-wq"), {})
        rpq = results.get((mode, "mhqg-pq"), {})
        b4w = rwq.get("Bleu_4", 0)
        rlw = rwq.get("ROUGE_L", 0)
        ovw = rwq.get("Overall", 0)
        b4p = rpq.get("Bleu_4", 0)
        rlp = rpq.get("ROUGE_L", 0)
        ovp = rpq.get("Overall", 0)
        print(f"{mode:<28} | {b4w:6.2f} {rlw:6.2f} {ovw:6.2f} | {b4p:6.2f} {rlp:6.2f} {ovp:6.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
