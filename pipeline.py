"""
OpenAgentQG two-stage pipeline: neuro-symbolic fusion -> agentic collaborative generation.
ONE_SHOT_Q + BATCH: single API call for multiple questions, parallel workers for speed.
"""
import time
from config import ONE_SHOT_Q, BATCH_QS_PER_CALL, ABLATION_MODE
from fusion import neuro_symbolic_fusion, build_fusion_graph
from agentic.agents import agentic_collaborative_generation, batch_one_shot_generate, one_shot_generate


def run_open_agent_qg(sample, verbose=True):
    """
    Run full OpenAgentQG for one sample.
    ONE_SHOT_Q: skip Stage1 LLM, single triples+answer->question call.
    """
    step_times = {}
    answers = sample.get("answers", []) or ["(unknown)"]
    triples = sample.get("triples", [])
    entities = sample.get("entities", set())

    no_fusion = ABLATION_MODE == "no_fusion"
    if ONE_SHOT_Q or no_fusion:
        t0 = time.time()
        fusion = build_fusion_graph(triples, entities, "", [])
        step_times["1_fusion"] = time.time() - t0
        if verbose:
            print(f"  [Step 1/2] fusion (no LLM)" + (" [ablation:no_fusion]" if no_fusion else "") + f" {step_times['1_fusion']:.2f}s")
    else:
        t0 = time.time()
        fusion = neuro_symbolic_fusion(sample)
        step_times["1_fusion"] = time.time() - t0
        if verbose:
            print(f"  [Step 1/2] neuro_symbolic_fusion {step_times['1_fusion']:.1f}s")

    if ABLATION_MODE == "no_agentic":
        t0 = time.time()
        answers_list = answers if isinstance(answers, (list, tuple)) else [answers]
        question = one_shot_generate(fusion, answers_list)
        step_times["2_agentic"] = time.time() - t0
        score, iters = 0, 1
        if verbose:
            print(f"  [Step 2/2] one_shot (ablation:no_agentic) {step_times['2_agentic']:.2f}s")
    else:
        t0 = time.time()
        question, score, iters = agentic_collaborative_generation(fusion, answers)
        step_times["2_agentic"] = time.time() - t0
        if verbose:
            print(f"  [Step 2/2] agentic_generation {step_times['2_agentic']:.1f}s (score={score}, iters={iters})")

    total = sum(step_times.values())
    if verbose:
        print(f"  => total {total:.1f}s, question: {(question or '')[:60]}...")
    return {"question": question, "score": score, "iterations": iters, "step_times": step_times}


def run_open_agent_qg_batch(samples, verbose=False):
    """
    Batch samples by BATCH_QS_PER_CALL; one API call per chunk. Reduces calls when ONE_SHOT_Q.
    Returns: list of {"question": str, "score": 0, "iterations": 1}
    """
    if not samples:
        return []
    fusions = []
    answers_list = []
    for s in samples:
        triples = s.get("triples", [])
        entities = s.get("entities", set())
        fusion = build_fusion_graph(triples, entities, "", [])
        fusions.append(fusion)
        answers_list.append(s.get("answers", []) or ["(unknown)"])

    if not ONE_SHOT_Q:
        return [run_open_agent_qg(s, verbose=False) for s in samples]

    questions = []
    batch_size = BATCH_QS_PER_CALL
    t0 = time.time()
    for start in range(0, len(samples), batch_size):
        chunk = list(zip(fusions[start:start + batch_size], answers_list[start:start + batch_size]))
        for attempt in range(2):
            try:
                questions.extend(batch_one_shot_generate(chunk))
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(5)
                else:
                    questions.extend([""] * len(chunk))
    if verbose:
        print(f"  [Batch] {len(samples)} questions in {(len(samples)+batch_size-1)//batch_size} call(s), {time.time()-t0:.1f}s")
    return [{"question": q, "score": 0, "iterations": 1} for q in questions]
