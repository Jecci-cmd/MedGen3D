#!/usr/bin/env python3
"""Create an nnU-Net v2 dataset with the frozen 500/50/100 patient split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


LABELS = {
    "background": 0, "aorta": 1, "gall_bladder": 2, "kidney_left": 3,
    "kidney_right": 4, "liver": 5, "pancreas": 6, "postcava": 7,
    "spleen": 8, "stomach": 9,
}


def read_rows(root: Path, split: str):
    path = root / f"processed/manifests/{split}.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Refusing to replace {destination}")
        return
    os.symlink(source.resolve(), destination)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path("data/AbdomenAtlas1.0Mini"))
    p.add_argument("--nnunet-raw",type=Path,required=True); p.add_argument("--dataset-id",type=int,default=501)
    p.add_argument("--nnunet-preprocessed",type=Path,required=True)
    a=p.parse_args(); name=f"Dataset{a.dataset_id:03d}_MedGen3DAbdomen9"; out=a.nnunet_raw/name
    train=read_rows(a.root,"train"); val=read_rows(a.root,"val"); test=read_rows(a.root,"test")
    for row in train+val:
        cid=row["case_id"]; link(a.root/row["image"],out/"imagesTr"/f"{cid}_0000.nii.gz"); link(a.root/row["mask"],out/"labelsTr"/f"{cid}.nii.gz")
    for row in test:
        cid=row["case_id"]; link(a.root/row["image"],out/"imagesTs"/f"{cid}_0000.nii.gz"); link(a.root/row["mask"],out/"labelsTs"/f"{cid}.nii.gz")
    payload={"channel_names":{"0":"CT"},"labels":LABELS,"numTraining":len(train)+len(val),"file_ending":".nii.gz","overwrite_image_reader_writer":"NibabelIOWithReorient"}
    (out/"dataset.json").write_text(json.dumps(payload,indent=2))
    pre=a.nnunet_preprocessed/name; pre.mkdir(parents=True,exist_ok=True)
    split=[{"train":[r["case_id"] for r in train],"val":[r["case_id"] for r in val]}]
    (pre/"splits_final.json").write_text(json.dumps(split,indent=2))
    print(json.dumps({"dataset":name,"train":len(train),"val":len(val),"test":len(test),"output":str(out)}))


if __name__=="__main__": main()
