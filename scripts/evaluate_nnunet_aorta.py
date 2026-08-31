#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import nibabel as nib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from medgen3d.evaluation import segmentation_metrics
from medgen3d.data import DynamicCaseDataset
from medicalmodel_data.geometry import crop_or_pad

def arr(p): return np.moveaxis(np.asanyarray(nib.load(p).dataobj),(0,1,2),(2,1,0))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--pred",type=Path,required=True); p.add_argument("--gt",type=Path,required=True); p.add_argument("--cases",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--root",type=Path,default=Path("data/AbdomenAtlas1.0Mini")); p.add_argument("--patch-shape",type=int,nargs=3,default=(97,96,96)); p.add_argument("--seed",type=int,default=20260812); a=p.parse_args()
    case_ids=[x.strip() for x in a.cases.read_text().splitlines() if x.strip()]
    dataset=DynamicCaseDataset(a.root,"processed/manifests/test.jsonl","test",a.patch_shape,evaluation_task="segmentation",segmentation_target="sdf",seed=a.seed,num_samples=len(case_ids),case_ids=case_ids)
    rows=[]
    for index in range(len(dataset)):
        sample=dataset[index]; cid=sample["case_id"]; pred_path=a.pred/f"{cid}.nii.gz"; image=nib.load(pred_path)
        spacing=tuple(float(x) for x in image.header.get_zooms()[:3][::-1]); start=tuple(sample["metadata"]["crop_start_zyx"])
        pred=crop_or_pad(arr(pred_path)==1,start,a.patch_shape,fill_value=False)
        target=sample["target"].numpy()[0] < 0; valid=sample["valid_mask"].numpy()[0].astype(bool)
        coords=np.where(valid); crop=tuple(slice(int(axis.min()),int(axis.max())+1) for axis in coords)
        metric=segmentation_metrics(pred[crop],target[crop],spacing)
        rows.append({"case_id":cid,"metrics":metric}); print(json.dumps(rows[-1]),flush=True)
    keys=rows[0]["metrics"]; summary={}
    for key in keys:
        values=np.asarray([r["metrics"][key] for r in rows],float); finite=values[np.isfinite(values)]
        summary[key]={"mean":float(finite.mean()) if len(finite) else float("inf"),"median":float(np.median(finite)) if len(finite) else float("inf"),"finite_cases":int(len(finite)),"infinite_cases":int((~np.isfinite(values)).sum())}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({"model":"nnUNet-v2-3d_fullres","organ":"aorta","num_cases":len(rows),"summary":summary,"rows":rows},indent=2))
if __name__=="__main__": main()
