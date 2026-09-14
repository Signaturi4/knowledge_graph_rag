"""Build a small, balanced evaluation cut from a PMC-Patients CSV prefix.

Input : eval/data/pmc_head.csv   (a byte-range prefix of
        https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients.csv)
Output: eval/data/pmc_cut.jsonl  ({patient_id, age, gender, text, condition_hint})

Kept deliberately small: enough patients to exercise the full pipeline, few
enough to run LLM extraction in a couple of minutes.
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "pmc_head.csv")
OUT = os.path.join(HERE, "data", "pmc_cut.jsonl")

csv.field_size_limit(10**7)

_COND = re.compile(
    r"\b(ARDS|COVID-19|pneumonia|carcinoma|lymphoma|leukaemia|leukemia|sepsis|stroke|"
    r"myocardial infarction|tuberculosis|diabetes|hypertension|melanoma|sarcoma|"
    r"appendicitis|pancreatitis|hepatitis|nephropathy|anemia|anaemia|fracture)\b",
    re.I,
)


def _age(raw: str) -> str:
    try:
        v = ast.literal_eval(raw)
        n, unit = v[0][0], v[0][1]
        return f"{int(n)} {unit}s" if n != 1 else f"1 {unit}"
    except Exception:
        return raw.strip()


def main(n: int = 40, min_chars: int = 300, max_chars: int = 2500) -> int:
    if not os.path.exists(SRC):
        print(f"missing {SRC} — download a CSV prefix first", file=sys.stderr)
        return 1
    rows_out = []
    with open(SRC, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {c: i for i, c in enumerate(header)}
        for row in r:
            if len(row) != len(header):
                continue
            text = row[idx["patient"]].strip()
            if not (min_chars <= len(text) <= max_chars):
                continue
            m = _COND.search(text)
            rows_out.append({
                "patient_id": row[idx["patient_id"]],
                "age": _age(row[idx["age"]]),
                "gender": {"M": "male", "F": "female"}.get(row[idx["gender"]].strip(),
                                                          row[idx["gender"]].strip()),
                "text": text,
                "condition_hint": (m.group(0).lower() if m else ""),
            })
            if len(rows_out) >= n:
                break

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    with_cond = sum(1 for r in rows_out if r["condition_hint"])
    print(f"wrote {len(rows_out)} patients -> {OUT}  ({with_cond} with a detected condition)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
