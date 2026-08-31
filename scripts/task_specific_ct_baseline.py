#!/usr/bin/env python3
"""Train/evaluate RED-CNN or FBPConvNet on frozen MedGen3D patient splits."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medgen3d.evaluation import paired_ct_metrics, summarize_paired_ct
from medgen3d.data import DynamicCaseDataset


def load_zyx(path: Path) -> np.ndarray:
    return np.moveaxis(np.asanyarray(nib.load(path).dataobj), (0, 1, 2), (2, 1, 0)).astype(np.float32)


def norm(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1000, 1000) / 1000.0


class REDCNN(nn.Module):
    def __init__(self, width: int = 96):
        super().__init__()
        self.enc = nn.ModuleList([nn.Conv2d(1 if i == 0 else width, width, 5) for i in range(5)])
        self.dec = nn.ModuleList([nn.ConvTranspose2d(width, width if i < 4 else 1, 5) for i in range(5)])

    def forward(self, x):
        residual, skips, h = x, [], x
        for layer in self.enc:
            h = F.relu(layer(h), inplace=True); skips.append(h)
        for i, layer in enumerate(self.dec):
            h = layer(h)
            if i < 4:
                h = F.relu(h + skips[-2-i], inplace=True)
        return residual + h


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class FBPConvNet(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.b1, self.b2, self.b3, self.b4 = ConvBlock(1,width), ConvBlock(width,width*2), ConvBlock(width*2,width*4), ConvBlock(width*4,width*8)
        self.pool = nn.MaxPool2d(2)
        self.u3, self.u2, self.u1 = nn.ConvTranspose2d(width*8,width*4,2,2), nn.ConvTranspose2d(width*4,width*2,2,2), nn.ConvTranspose2d(width*2,width,2,2)
        self.d3, self.d2, self.d1 = ConvBlock(width*8,width*4), ConvBlock(width*4,width*2), ConvBlock(width*2,width)
        self.head = nn.Conv2d(width,1,1)

    def forward(self, x):
        a=self.b1(x); b=self.b2(self.pool(a)); c=self.b3(self.pool(b)); d=self.b4(self.pool(c))
        h=self.d3(torch.cat([self.u3(d),c],1)); h=self.d2(torch.cat([self.u2(h),b],1)); h=self.d1(torch.cat([self.u1(h),a],1))
        return x + self.head(h)


def records(root: Path, split: str):
    p=root/f"processed/manifests/{split}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def source_key(model: str) -> str:
    return "ldct" if model == "redcnn" else "sparse_view_ct"


def get_pair(root: Path, row: dict, model: str):
    src = row[source_key(model)][0]
    return load_zyx(root/src), load_zyx(root/row["image"])


def crop_pair(x, y, size, rng):
    z=int(rng.integers(x.shape[0])); h,w=x.shape[1:]
    y0=int(rng.integers(max(1,h-size+1))); x0=int(rng.integers(max(1,w-size+1)))
    a=x[z,y0:y0+size,x0:x0+size]; b=y[z,y0:y0+size,x0:x0+size]
    ph,pw=size-a.shape[0],size-a.shape[1]
    if ph or pw: a=np.pad(a,((0,ph),(0,pw)),constant_values=-1000); b=np.pad(b,((0,ph),(0,pw)),constant_values=-1000)
    return norm(a),norm(b)


def build(name):
    return REDCNN() if name == "redcnn" else FBPConvNet()


@torch.no_grad()
def val_loss(model, root, rows, name, device, seed=0):
    rng=np.random.default_rng(seed); values=[]; model.eval()
    for row in rows[:20]:
        x,y=get_pair(root,row,name); a,b=crop_pair(x,y,192,rng)
        pred=model(torch.from_numpy(a)[None,None].to(device)).float().cpu().numpy()[0,0]
        values.append(float(np.mean(np.abs(pred-b))))
    return float(np.mean(values))


def train(args):
    root=args.root; train_rows=records(root,"train"); val_rows=records(root,"val")
    random.seed(args.seed); np_rng=np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda"); model=build(args.model).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scaler=torch.amp.GradScaler("cuda"); args.output.mkdir(parents=True,exist_ok=True)
    best=float("inf"); order=np_rng.permutation(len(train_rows)); order_at=0; pair=None; pair_uses=0
    for step in range(1,args.steps+1):
        if pair is None or pair_uses >= args.steps_per_volume:
            if order_at >= len(order): order=np_rng.permutation(len(train_rows)); order_at=0
            pair=get_pair(root,train_rows[int(order[order_at])],args.model); order_at+=1; pair_uses=0
        pair_uses += 1
        xs=[]; ys=[]
        for _ in range(args.batch_size):
            a,b=crop_pair(*pair,args.crop_size,np_rng); xs.append(a); ys.append(b)
        x=torch.from_numpy(np.stack(xs))[:,None].to(device); y=torch.from_numpy(np.stack(ys))[:,None].to(device)
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): pred=model(x); loss=F.l1_loss(pred,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step%100==0: print(json.dumps({"step":step,"train_l1":float(loss),"model":args.model}),flush=True)
        if step%args.val_every==0 or step==args.steps:
            score=val_loss(model,root,val_rows,args.model,device,args.seed)
            state={"model":model.state_dict(),"step":step,"val_l1":score,"args":vars(args)}
            torch.save(state,args.output/"last.pt")
            if score<best: best=score; torch.save(state,args.output/"best.pt")
            print(json.dumps({"step":step,"val_l1":score,"best":best}),flush=True)


@torch.no_grad()
def predict_volume(model, x, device, batch=16):
    h,w=x.shape[1:]; ph=(-h)%16; pw=(-w)%16; out=[]; model.eval()
    for start in range(0,len(x),batch):
        a=norm(x[start:start+batch]); a=np.pad(a,((0,0),(0,ph),(0,pw)),constant_values=-1)
        p=model(torch.from_numpy(a)[:,None].to(device)).float().cpu().numpy()[:,0,:h,:w]
        out.append(p)
    return np.concatenate(out)*1000.0


def evaluate(args):
    case_ids=[x.strip() for x in args.cases.read_text().splitlines() if x.strip()]
    task="restoration" if args.model == "redcnn" else "reconstruction"
    dataset=DynamicCaseDataset(
        args.root, "processed/manifests/test.jsonl", "test", args.patch_shape,
        evaluation_task=task, segmentation_target="sdf", hu_clip=(-1000.0,1000.0),
        output_range=(-1.0,1.0), seed=args.eval_seed, num_samples=len(case_ids),
        case_ids=case_ids,
    )
    model=build(args.model).cuda(); state=torch.load(args.checkpoint,map_location="cpu",weights_only=False); model.load_state_dict(state["model"])
    output=[]
    for i in range(len(dataset)):
        sample=dataset[i]
        valid=sample["valid_mask"].numpy()[0].astype(bool)
        crop=tuple(slice(int(a.min()),int(a.max())+1) for a in np.where(valid))
        x=(sample["condition"].numpy()[0]+1.0)*1000.0
        y=(sample["target"].numpy()[0]+1.0)*1000.0
        pred=predict_volume(model,x,torch.device("cuda"),args.eval_batch)
        metric=paired_ct_metrics(x[crop],pred[crop],y[crop])
        output.append({"case_id":sample["case_id"],"metrics":metric})
        print(json.dumps({"done":i+1,"total":len(dataset),"case_id":sample["case_id"],"metrics":metric}),flush=True)
    payload={"model":args.model,"checkpoint":str(args.checkpoint),"cases":len(output),"rows":output,
             "summary":summarize_paired_ct([r["metrics"] for r in output],seed=args.seed)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["train","eval"]); p.add_argument("--model",choices=["redcnn","fbpconvnet"],required=True)
    p.add_argument("--root",type=Path,default=Path("data/AbdomenAtlas1.0Mini")); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path); p.add_argument("--cases",type=Path,default=Path("analysis/eval100_protocol/case_ids.txt"))
    p.add_argument("--steps",type=int,default=20000); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--crop-size",type=int,default=192)
    p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--val-every",type=int,default=1000); p.add_argument("--eval-batch",type=int,default=16); p.add_argument("--seed",type=int,default=20260813)
    p.add_argument("--steps-per-volume",type=int,default=16)
    p.add_argument("--patch-shape",type=int,nargs=3,default=(97,96,96)); p.add_argument("--eval-seed",type=int,default=20260812)
    args=p.parse_args(); train(args) if args.mode=="train" else evaluate(args)


if __name__ == "__main__": main()
