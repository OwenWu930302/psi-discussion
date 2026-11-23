"""
為 failure_diagnostics.py 準備 .npz：BUSI 測試集（官方切分的 val，130 張）
  probs (N,256,256) 機率圖（sigmoid）；gts 標註（256×256，最近鄰，> 127）；preds = probs ≥ 0.5（同 predict_busi.py）
  cases (N,) 影像名稱（failure_diagnostics 不讀；用來把 per_case.csv 的編號對回影像）
  --features：features (N,3) = psi1_convex、psi6_cc、psi8_rotation，取自 baseline 的 percase_best.csv（用 baseline 的預測算的 Ψ）
  --grads：每張影像的梯度（failure_diagnostics.extract_grads；loss = 訓練用的 BCEDiceLoss；參數 = 輸出層 final 的權重與偏差）
           grads_eval = 測試集 130 張；grads_flag = Ψ8 加權清單中的訓練影像（155 張），weights_flag = --wfactor；
           grads_ctrl = 其餘訓練影像（362 張，對照）
用法（Seg_UKAN 資料夾，ukan 環境）：
  python make_diag_npz.py --name busi_UKAN_seed2981 --which best --features --grads
  python make_diag_npz.py --name busi_UKAN_base2 --which last
"""
import argparse
import importlib.util
import os
import sys
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations import Compose, Resize
from albumentations.augmentations import transforms
from sklearn.model_selection import train_test_split

import archs
import losses

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--which', default='last', choices=['best', 'last'])
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--features', action='store_true')
ap.add_argument('--feature_csv', default='outputs/busi_UKAN_seed2981/percase_best.csv')
ap.add_argument('--grads', action='store_true')
ap.add_argument('--flag_list', default='/mnt/data2t/BUSI/psi_lists/wcases_busi_psi8_rotation.csv')
ap.add_argument('--wfactor', default=3.0, type=float)
ap.add_argument('--params', default='final', help='算梯度的參數：名稱以這些開頭者，逗號分隔（預設 final = 輸出層；例 decoder5,final）')
ap.add_argument('--diag', default='failure_diagnostics.py')
ap.add_argument('--out', default=None)
args = ap.parse_args()
FEATS = ['psi1_convex', 'psi6_cc', 'psi8_rotation']

D = f'{args.output_dir}/{args.name}'
config = yaml.load(open(f'{D}/config.yml'), Loader=yaml.FullLoader)
ck = f'{D}/model.pth' if args.which == 'best' else f'{D}/model_last.pth'
os.path.exists(ck) or sys.exit(f'找不到 {ck}')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'],
                                       embed_dims=config['input_list'], no_kan=config.get('no_kan', False))
model.load_state_dict(torch.load(ck, map_location='cpu'), strict=True)
model.to(dev).eval()
print(f'模型：{ck}')

base = os.path.join(config['data_dir'], config['dataset'])
ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(base, 'images', '*.png')))
tr, va = train_test_split(ids, test_size=0.2, random_state=config['dataseed'])     # 同 train.py / predict_busi.py
if config.get('move_to_train'):
    mv = set(pd.read_csv(config['move_to_train'], encoding='utf-8-sig').iloc[:, 0].astype(str))
    tr, va = tr + [i for i in va if i in mv], [i for i in va if i not in mv]
tf = Compose([Resize(config['input_h'], config['input_w']), transforms.Normalize()])
H, W = config['input_h'], config['input_w']


def load(i):
    img = cv2.imread(os.path.join(base, 'images', i + '.png'))
    gt = cv2.imread(os.path.join(base, 'masks', '0', i + '_mask.png'), cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy((tf(image=img)['image'].astype('float32') / 255).transpose(2, 0, 1))[None]
    g = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST) > 127
    return x, g


out = {'cases': np.array(va)}
probs, gts = [], []
with torch.no_grad():
    for i in va:
        x, g = load(i)
        probs.append(torch.sigmoid(model(x.to(dev)))[0, 0].cpu().numpy().astype(np.float32))
        gts.append(g)
out['probs'], out['gts'] = np.stack(probs), np.stack(gts)
out['preds'] = out['probs'] >= 0.5
dd = [2 * (p & g).sum() / max(p.sum() + g.sum(), 1) for p, g in zip(out['preds'], out['gts'])]
print(f'測試集 {len(va)} 張，平均 Dice {np.mean(dd):.4f}，Dice < 0.5 有 {int((np.array(dd) < 0.5).sum())} 張')

if args.features:
    f = pd.read_csv(args.feature_csv).set_index('case')
    out['features'] = f.loc[va, FEATS].to_numpy(float)
    out['feature_names'] = np.array(FEATS)
    print(f'Ψ 特徵：{FEATS}（{args.feature_csv}）')

if args.grads:
    spec = importlib.util.spec_from_file_location('fd', args.diag)
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    flag = set(pd.read_csv(args.flag_list, encoding='utf-8-sig').iloc[:, 0].astype(str))
    fl = [i for i in tr if i in flag]
    ct = [i for i in tr if i not in flag]
    PFX = tuple(args.params.split(','))
    names = [n for n, p in model.named_parameters() if p.requires_grad and n.startswith(PFX)]
    print(f'梯度參數：{names}（{sum(p.numel() for n, p in model.named_parameters() if n in names)} 維）')
    lossf = losses.BCEDiceLoss()

    def loader(lst):
        for i in lst:
            x, g = load(i)
            yield x, torch.from_numpy(g.astype(np.float32))[None, None]

    grads = lambda lst: fd.extract_grads(model, lambda o, y: lossf(o, y), loader(lst), device=dev,
                                         param_filter=lambda n: n.startswith(PFX))
    out['grads_eval'] = grads(va)
    out['grads_flag'] = grads(fl)
    out['weights_flag'] = np.full(len(fl), args.wfactor)
    out['grads_ctrl'] = grads(ct)
    print(f'梯度：測試 {len(va)}、加權清單 {len(fl)}、對照 {len(ct)}')

path = args.out or f'{args.output_dir}/diag/{args.name}_{args.which}.npz'
os.makedirs(os.path.dirname(path), exist_ok=True)
np.savez_compressed(path, **out)
print(f'→ {path}')