#!/usr/bin/env python3
"""Build auditable JSON/CSV/Markdown comparisons for the frozen 100 cases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path: Path):
    return json.loads(path.read_text())


def mean_ci(values: list[float], seed: int = 20260813):
    x=np.asarray(values,float); rng=np.random.default_rng(seed)
    boot=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1)
    return {"mean":float(x.mean()),"median":float(np.median(x)),"ci95":[float(v) for v in np.percentile(boot,[2.5,97.5])]}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path("analysis/task_specific")); a=p.parse_args()
    med=load(a.root/"medgen3d_eval100/results.json")
    sources={
        "segmentation":load(a.root/"nnunet_eval100.json"),
        "restoration":load(a.root/"redcnn/eval100.json"),
        "reconstruction":load(a.root/"fbpconvnet/eval100.json"),
    }
    med_rows={t:{r["case_id"]:r for r in med["rows"] if r["task"]==t} for t in sources}
    spec_rows={t:{r["case_id"]:r for r in payload["rows"]} for t,payload in sources.items()}
    table=[]; result={"protocol":"analysis/eval100_protocol/README.md","num_cases":100,"tasks":{}}
    for task in sources:
        common=sorted(set(med_rows[task]) & set(spec_rows[task]))
        if len(common)!=100: raise RuntimeError(f"{task}: expected 100 paired cases, found {len(common)}")
        metrics=(sorted(med_rows[task][common[0]]["metrics"])
                 if task=="segmentation" else sorted(med_rows[task][common[0]]["metrics"]["model"]))
        task_result={"specific_model":{"segmentation":"nnU-Net v2 3d_fullres","restoration":"RED-CNN","reconstruction":"FBPConvNet"}[task],"metrics":{}}
        for metric in metrics:
            if task=="segmentation":
                mv=[float(med_rows[task][c]["metrics"][metric]) for c in common]
                sv=[float(spec_rows[task][c]["metrics"][metric]) for c in common]
            else:
                mv=[float(med_rows[task][c]["metrics"]["model"][metric]) for c in common]
                sv=[float(spec_rows[task][c]["metrics"]["model"][metric]) for c in common]
            delta=(np.asarray(mv)-np.asarray(sv)).tolist()
            task_result["metrics"][metric]={"medgen3d":mean_ci(mv),"specific":mean_ci(sv),"paired_delta_medgen_minus_specific":mean_ci(delta)}
            for c,m,s,d in zip(common,mv,sv,delta): table.append({"case_id":c,"task":task,"metric":metric,"medgen3d":m,"specific":s,"medgen_minus_specific":d})
        result["tasks"][task]=task_result
    (a.root/"comparison.json").write_text(json.dumps(result,indent=2))
    with (a.root/"per_case_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=table[0]); w.writeheader(); w.writerows(table)
    lines=["# Frozen eval100: MedGen3D vs task-specific models","", "All values are paired over the same 100 cases and the same valid 97×96×96 evaluation patch.",""]
    for task,payload in result["tasks"].items():
        lines += [f"## {task} — {payload['specific_model']}","", "| Metric | MedGen3D mean | Specific mean | Paired delta |","|---|---:|---:|---:|"]
        for metric,v in payload["metrics"].items(): lines.append(f"| {metric} | {v['medgen3d']['mean']:.6f} | {v['specific']['mean']:.6f} | {v['paired_delta_medgen_minus_specific']['mean']:.6f} |")
        lines.append("")
    (a.root/"comparison.md").write_text("\n".join(lines))


if __name__=="__main__": main()
