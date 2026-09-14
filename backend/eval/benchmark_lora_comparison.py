import os
import sys
import json
import ast
import re
import time
from typing import List, Tuple, Set, Dict, Any

WORKSPACE_ROOT = "/Users/signatur4ik/Desktop/Determenistic_memory_layer"
NB_DIR = os.path.join(WORKSPACE_ROOT, "knowledge_graph_rag/GTC25_DLI/notebooks")
DATA_DIR = os.path.join(WORKSPACE_ROOT, "knowledge_graph_rag/GTC25_DLI/data/training_data")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")
LORA_DIR = os.path.join(WORKSPACE_ROOT, "knowledge_graph_rag/GTC25_DLI/model/loras/Meta-Llama-3-8B-Instruct-PMC-LoRA-mlx")
BASE_MODEL_NAME = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

STANDARD_PREDICATES = {"has_age", "has_gender", "diagnosed_with", "presents_with", "receives"}

def extract_triplets(raw_output: str) -> Tuple[bool, bool, List[Tuple]]:
    """
    Parse LLM output into triplets.
    Returns:
      is_strict_python: bool (True if output is clean Python list without conversational text)
      is_schema_compliant: bool (True if tuples have 5 elements and standard ontology)
      triplets: List of 5-tuples
    """
    text = raw_output.strip()
    if "<|eot_id|>" in text:
        text = text.split("<|eot_id|>")[0].strip()

    is_strict_python = False
    # Test if entire text is a valid Python list literal (no markdown fences, no conversational prose)
    try:
        val = ast.literal_eval(text)
        if isinstance(val, list):
            is_strict_python = True
    except Exception:
        pass

    # Fallback to extract bracketed list if conversational filler exists
    content_to_parse = text
    if not is_strict_python:
        if "```python" in text:
            content_to_parse = text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            content_to_parse = text.split("```")[1].split("```")[0].strip()
        else:
            s = text.find("[")
            e = text.rfind("]")
            if s != -1 and e != -1 and e > s:
                content_to_parse = text[s:e+1]

    parsed_list = []
    try:
        val = ast.literal_eval(content_to_parse)
        if isinstance(val, list):
            parsed_list = val
    except Exception:
        pass

    # Validate 5-tuple schema compliance: (Subject, Subj_Type, Predicate, Object, Obj_Type)
    valid_tuples = []
    schema_compliant_count = 0
    for item in parsed_list:
        if isinstance(item, (list, tuple)) and len(item) == 5:
            subj, stype, pred, obj, otype = [str(x).strip() for x in item]
            valid_tuples.append((subj, stype, pred, obj, otype))
            if pred.lower() in STANDARD_PREDICATES:
                schema_compliant_count += 1
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            # 3-tuple format fallback
            s, p, o = [str(x).strip() for x in item]
            valid_tuples.append((s, "UNKNOWN", p, o, "UNKNOWN"))

    is_schema_compliant = (len(valid_tuples) > 0 and schema_compliant_count == len(valid_tuples))
    return is_strict_python, is_schema_compliant, valid_tuples

def normalize_for_semantic_match(triplets: List[Tuple]) -> Set[Tuple]:
    """Normalize subject 'Patient <id>' -> 'patient' to isolate clinical entity/relation accuracy."""
    res = set()
    for item in triplets:
        if len(item) >= 5:
            subj = "patient"
            stype = str(item[1]).strip().lower()
            pred = str(item[2]).strip().lower()
            obj = str(item[3]).strip().lower()
            otype = str(item[4]).strip().lower()
            res.add((subj, stype, pred, obj, otype))
    return res

def calc_p_r_f1(pred_set: Set[Tuple], gt_set: Set[Tuple]) -> Dict[str, float]:
    if not pred_set and not gt_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(gt_set)}
    if not gt_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "tp": 0, "fp": len(pred_set), "fn": 0}

    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def run_benchmark(num_samples: int = 10):
    from mlx_lm import load, generate

    print(f"Reading test data from: {TEST_FILE}")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()][:num_samples]
    print(f"Total test samples to evaluate: {len(samples)}")

    # 1. BASE MODEL (WITHOUT LORA)
    print("\n" + "="*70)
    print("RUNNING INFERENCE: BASE MODEL (WITHOUT LORA)")
    print("="*70)
    t0 = time.time()
    base_model, base_tok = load(BASE_MODEL_NAME)
    print(f"Loaded {BASE_MODEL_NAME} in {time.time() - t0:.2f}s")

    base_results = []
    for i, s in enumerate(samples):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Extract all clinical entity-relation-entity triplets from the following patient case as a valid Python list:\n\n{s['input']}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        t_gen = time.time()
        output = generate(base_model, base_tok, prompt=prompt, max_tokens=256, verbose=False)
        dt = time.time() - t_gen

        is_strict, is_schema, pred_triplets = extract_triplets(output)
        _, _, gt_triplets = extract_triplets(s["output"])

        pred_norm = normalize_for_semantic_match(pred_triplets)
        gt_norm = normalize_for_semantic_match(gt_triplets)
        metrics = calc_p_r_f1(pred_norm, gt_norm)

        base_results.append({
            "idx": i,
            "patient_id": s.get("patient_id", str(i)),
            "is_strict": is_strict,
            "is_schema": is_schema,
            "output": output,
            "pred_count": len(pred_triplets),
            "gt_count": len(gt_triplets),
            "metrics": metrics,
            "time": dt
        })
        print(f"  Sample {i+1:02d} | Strict Python: {str(is_strict):5s} | Schema: {str(is_schema):5s} | P: {metrics['precision']:.2f} | R: {metrics['recall']:.2f} | F1: {metrics['f1']:.2f} ({dt:.2f}s)")

    del base_model
    del base_tok
    import gc
    gc.collect()

    # 2. FINE-TUNED MODEL (WITH LORA)
    print("\n" + "="*70)
    print("RUNNING INFERENCE: FINE-TUNED MODEL (WITH LORA)")
    print("="*70)
    t0 = time.time()
    lora_model, lora_tok = load(BASE_MODEL_NAME, adapter_path=LORA_DIR)
    print(f"Loaded {BASE_MODEL_NAME} + LoRA in {time.time() - t0:.2f}s")

    lora_results = []
    for i, s in enumerate(samples):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Extract all clinical entity-relation-entity triplets from the following patient case as a valid Python list:\n\n{s['input']}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        t_gen = time.time()
        output = generate(lora_model, lora_tok, prompt=prompt, max_tokens=256, verbose=False)
        dt = time.time() - t_gen

        is_strict, is_schema, pred_triplets = extract_triplets(output)
        _, _, gt_triplets = extract_triplets(s["output"])

        pred_norm = normalize_for_semantic_match(pred_triplets)
        gt_norm = normalize_for_semantic_match(gt_triplets)
        metrics = calc_p_r_f1(pred_norm, gt_norm)

        lora_results.append({
            "idx": i,
            "patient_id": s.get("patient_id", str(i)),
            "is_strict": is_strict,
            "is_schema": is_schema,
            "output": output,
            "pred_count": len(pred_triplets),
            "gt_count": len(gt_triplets),
            "metrics": metrics,
            "time": dt
        })
        print(f"  Sample {i+1:02d} | Strict Python: {str(is_strict):5s} | Schema: {str(is_schema):5s} | P: {metrics['precision']:.2f} | R: {metrics['recall']:.2f} | F1: {metrics['f1']:.2f} ({dt:.2f}s)")

    # 3. AGGREGATE SUMMARY
    N = len(samples)
    b_strict_pct = sum(1 for r in base_results if r["is_strict"]) / N * 100
    l_strict_pct = sum(1 for r in lora_results if r["is_strict"]) / N * 100

    b_schema_pct = sum(1 for r in base_results if r["is_schema"]) / N * 100
    l_schema_pct = sum(1 for r in lora_results if r["is_schema"]) / N * 100

    b_prec = sum(r["metrics"]["precision"] for r in base_results) / N
    l_prec = sum(r["metrics"]["precision"] for r in lora_results) / N

    b_rec = sum(r["metrics"]["recall"] for r in base_results) / N
    l_rec = sum(r["metrics"]["recall"] for r in lora_results) / N

    b_f1 = sum(r["metrics"]["f1"] for r in base_results) / N
    l_f1 = sum(r["metrics"]["f1"] for r in lora_results) / N

    b_time = sum(r["time"] for r in base_results) / N
    l_time = sum(r["time"] for r in lora_results) / N

    print("\n" + "="*75)
    print("FINAL QUANTITATIVE BENCHMARK: BASE MODEL vs. LORA FINE-TUNED")
    print("="*75)
    print(f"| Metric                              | Without LoRA (Base) | With LoRA (Fine-Tuned) | Absolute Delta |")
    print(f"|-------------------------------------|---------------------|------------------------|----------------|")
    print(f"| Strict Python List Parse Rate       | {b_strict_pct:18.1f}% | {l_strict_pct:21.1f}% | {l_strict_pct - b_strict_pct:+13.1f}% |")
    print(f"| KG Schema Compliance (5-tuple)      | {b_schema_pct:18.1f}% | {l_schema_pct:21.1f}% | {l_schema_pct - b_schema_pct:+13.1f}% |")
    print(f"| Triplet Semantic Precision          | {b_prec:19.3f} | {l_prec:22.3f} | {l_prec - b_prec:+14.3f} |")
    print(f"| Triplet Semantic Recall             | {b_rec:19.3f} | {l_rec:22.3f} | {l_rec - b_rec:+14.3f} |")
    print(f"| Triplet Semantic F1 Score           | {b_f1:19.3f} | {l_f1:22.3f} | {l_f1 - b_f1:+14.3f} |")
    print(f"| Average Inference Latency           | {b_time:17.2f}s | {l_time:20.2f}s | {l_time - b_time:+13.2f}s |")
    print("="*75)

    return {
        "base_results": base_results,
        "lora_results": lora_results,
        "summary": {
            "b_strict_pct": b_strict_pct, "l_strict_pct": l_strict_pct,
            "b_schema_pct": b_schema_pct, "l_schema_pct": l_schema_pct,
            "b_prec": b_prec, "l_prec": l_prec,
            "b_rec": b_rec, "l_rec": l_rec,
            "b_f1": b_f1, "l_f1": l_f1,
            "b_time": b_time, "l_time": l_time,
        }
    }

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_benchmark(num_samples=n)
