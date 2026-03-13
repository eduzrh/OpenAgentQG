"""
Load KGQG datasets (mhqg-wq / mhqg-pq style JSON) and extract subgraphs as triples.
Unsupervised: train/dev/test may omit outSeq; gold in eval_gold/<split>.txt for evaluation.
"""
import os
import json


def inGraph_to_triples(inGraph):
    """
    Convert inGraph (adjacency with node/edge names) to list of (subject, relation, object) triples.
    Returns: list of (s_name, r_name, o_name), entities set.
    """
    names = inGraph.get("g_node_names", {})
    adj = inGraph.get("g_adj", {})
    triples = []
    entities = set()
    edge_types = inGraph.get("g_edge_types", {})
    for s_id, neighbors in adj.items():
        s_name = names.get(s_id, str(s_id))
        if isinstance(s_name, list):
            s_name = s_name[0] if s_name else str(s_id)
        entities.add(s_name)
        for o_id, r_id_or_list in neighbors.items():
            o_name = names.get(o_id, str(o_id))
            if isinstance(o_name, list):
                o_name = o_name[0] if o_name else str(o_id)
            r_id = r_id_or_list[0] if isinstance(r_id_or_list, list) else r_id_or_list
            r_name = edge_types.get(r_id, r_id)
            if isinstance(r_name, dict):
                r_name = list(r_name.values())[0] if r_name else str(r_id)
            if isinstance(r_name, list):
                r_name = r_name[0] if r_name else str(r_id)
            triples.append((s_name, r_name, o_name))
            entities.add(o_name)
    return triples, entities


def load_mhqg_json(path, max_samples=None):
    """
    Load one JSONL or JSON array file. Each line (or each item) is a sample.
    Returns: list of {
        "qId", "answers", "answer_ids", "outSeq",
        "triples": [(s,r,o)], "entities": set, "inGraph" (original)
    }
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inGraph = obj.get("inGraph", {})
            triples, entities = inGraph_to_triples(inGraph)
            samples.append({
                "qId": obj.get("qId", i),
                "answers": obj.get("answers", []),
                "answer_ids": obj.get("answer_ids", []),
                "outSeq": obj.get("outSeq", ""),
                "triples": triples,
                "entities": entities,
                "inGraph": inGraph,
            })
    return samples


def load_gold_for_eval(data_dir, dataset, split):
    """
    Load gold questions for evaluation. Returns list[str] aligned with split order, or None if missing.
    """
    path = os.path.join(data_dir, dataset, "eval_gold", f"{split}.txt")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line is not None]


def triples_to_text(triples, max_triples=20):
    """Format triples for LLM input."""
    lines = []
    for s, r, o in triples[:max_triples]:
        lines.append(f"({s}, {r}, {o})")
    return "\n".join(lines) if lines else "(no triples)"
