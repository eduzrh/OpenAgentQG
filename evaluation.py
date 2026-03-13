"""
"""
import sys
import os
from collections import defaultdict

_OPENAGENT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _OPENAGENT_ROOT not in sys.path:
    sys.path.insert(0, _OPENAGENT_ROOT)

_HAS_QGEVAL = False
try:
    from core.evaluation.eval import QGEvalCap
    _HAS_QGEVAL = True
except Exception:
    try:
        from config import GRAPH2SEQ_ROOT
        _g2s_src = os.path.join(GRAPH2SEQ_ROOT, "src")
        if os.path.isdir(_g2s_src) and _g2s_src not in sys.path:
            sys.path.insert(0, _g2s_src)
        from core.evaluation.eval import QGEvalCap
        _HAS_QGEVAL = True
    except Exception:
        pass

# Metric order aligned with Graph2Seq
_METRIC_ORDER = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L"]


def compute_metrics(gold_questions, pred_questions):
    """
    Compute metrics. QGEvalCap returns [0,1]; we scale to 0-100.
    """
    assert len(gold_questions) == len(pred_questions)
    gts = defaultdict(list)
    res = defaultdict(list)
    for i in range(len(gold_questions)):
        gts[i] = [str(gold_questions[i]).strip()]
        res[i] = [str(pred_questions[i]).strip()]

    if _HAS_QGEVAL:
        QGEval = QGEvalCap(gts, res)
        scores = QGEval.evaluate(verbose=False)
    else:
        scores = {"Bleu_1": 0.0, "Bleu_2": 0.0, "Bleu_3": 0.0, "Bleu_4": 0.0, "ROUGE_L": 0.0}

    # Scale [0,1] to 0-100
    for k in _METRIC_ORDER:
        v = scores.get(k, 0.0)
        if v <= 1.0:
            v = v * 100.0
        scores[k] = v
    scores["Overall"] = (scores["Bleu_4"] + scores["ROUGE_L"]) / 2.0
    return scores


def format_like_graph2seq(test_exs, metrics, step=1, total_steps=1):
    """
    Format output like Graph2Seq: [test] | test_exs = N | step: [x/y] | BLEU_1 = ... | ...
    """
    parts = ["[test] | test_exs = {} | step: [{} / {}]".format(test_exs, step, total_steps)]
    for k in _METRIC_ORDER:
        v = metrics.get(k, 0.0)
        if v <= 1.0:
            v = v * 100.0
        parts.append(" {} = {:0.3f}".format(k.upper(), v))
    return " |".join(parts)
