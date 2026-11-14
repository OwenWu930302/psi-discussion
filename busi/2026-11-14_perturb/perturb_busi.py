"""
BUSI 擾動測試：搬移的 9 張影像，模型「看過」之後是真的學會，還是只記住原圖？
  影像與遮罩做相同的變化 → 模型預測 → 與變化後的遮罩算 Dice（256×256，同 predict_busi.py）
  模型：busi_UKAN_moved（9 張在訓練集，看過）、busi_UKAN_base2（9 張在測試集，沒看過），預設 last
  變化分兩類：
    seen  ：官方 train.py 的資料增強本來就有（RandomRotate90、Flip）→ 訓練時看過，只作流程檢查
    novel ：訓練時從未出現 → shift 平移、scale 縮放、rotate 小角度旋轉、brightness 亮度、contrast 對比、
            gamma、blur 高斯模糊、speckle 乘性斑點雜訊；各 3 個強度（strength 1 輕 / 2 中 / 3 重）
  幾何變化：影像線性內插、邊緣補影像最小值（同 Ψ8）；遮罩最近鄰
  none（不變化）必須與 outputs/<模型>/percase_<last|best>.csv 的 Dice 完全相同（程式會自動比對）
輸出（outputs/analysis_busi_perturb/）：
  perturb_percase.csv  每個模型 × 影像 × 變化一列
  perturb_summary.csv  每個模型 × 變化 × 強度的平均（方向 / 正負合併）
  png/<影像>.png       原圖與幾個代表性變化的預測輪廓（綠 = 標註、紅 = 預測）
用法（在 Seg_UKAN 資料夾，ukan 環境；只做推論，不訓練）：
  python perturb_busi.py
  python perturb_busi.py --ref 30        # 另加 30 張一般訓練影像，只用來看「變化本身」對任何影像的影響
"""
import argparse
import os
import sys
import zlib
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

ap = argparse.ArgumentParser()
ap.add_argument('--models', default='busi_UKAN_moved,busi_UKAN_base2')
ap.add_argument('--which', default='last', choices=['best', 'last'])
ap.add_argument('--list', default='/mnt/data2t/BUSI/psi_lists/moved_busi_dice02.csv')
ap.add_argument('--ref', default=0, type=int, help='另加幾張一般訓練影像（兩個模型都看過）；0 = 不加')
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--out', default='outputs/analysis_busi_perturb')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = ap.parse_args()

# ---------- 變化的定義：(perturb, strength, level, variant) ----------
CONDS = [('none', 0, 0, '')]
CONDS += [('seen', 0, 0, v) for v in ['hflip', 'vflip', 'rot90', 'rot180']]
for s, l in enumerate([0.05, 0.10, 0.20], 1):
    CONDS += [('shift', s, l, v) for v in ['left', 'right', 'up', 'down']]
for s, l in enumerate([0.10, 0.20, 0.30], 1):
    CONDS += [('scale', s, l, v) for v in ['in', 'out']]          # 縮放倍率 1+l / 1-l
for s, l in enumerate([10, 20, 30], 1):
    CONDS += [('rotate', s, l, v) for v in ['+', '-']]
for s, l in enumerate([15, 30, 50], 1):
    CONDS += [('brightness', s, l, v) for v in ['+', '-']]        # 像素值 ± l（0–255）
for s, l in enumerate([0.2, 0.4, 0.6], 1):
    CONDS += [('contrast', s, l, v) for v in ['+', '-']]          # 對比倍率 1+l / 1-l
for s, l in enumerate([0.25, 0.5, 1.0], 1):
    CONDS += [('gamma', s, l, v) for v in ['+', '-']]             # gamma = 1+l / 1/(1+l)
for s, l in enumerate([1.0, 2.0, 3.0], 1):
    CONDS += [('blur', s, l, '')]                                 # 高斯 σ（像素）
for s, l in enumerate([0.1, 0.2, 0.3], 1):
    CONDS += [('speckle', s, l, '')]                              # img × (1 + l·N(0,1))
PNG_CONDS = [('none', 0, ''), ('seen', 0, 'hflip'), ('shift', 2, 'right'), ('scale', 2, 'out'), ('rotate', 2, '+'),
             ('contrast', 2, '-'), ('blur', 2, ''), ('speckle', 2, '')]


def affine(img, msk, M):
    h, w = msk.shape
    border = tuple(int(v) for v in img.reshape(-1, 3).min(0))
    i = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    m = cv2.warpAffine(msk.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return i, m > 0


def perturb(img, msk, p, l, v, case):
    """img：256×256×3 uint8；msk：256×256 bool → 回傳變化後的 (img, msk)"""
    h, w = msk.shape
    c = (w / 2, h / 2)
    if p == 'none':
        return img, msk
    if p == 'seen':
        f = {'hflip': lambda a: np.flip(a, 1), 'vflip': lambda a: np.flip(a, 0),          # 同 cv2.flip（OpenCV 4 不收 bool）
             'rot90': lambda a: np.rot90(a, 1), 'rot180': lambda a: np.rot90(a, 2)}[v]
        return np.ascontiguousarray(f(img)), np.ascontiguousarray(f(msk))
    if p == 'shift':
        dx, dy = {'left': (-l * w, 0), 'right': (l * w, 0), 'up': (0, -l * h), 'down': (0, l * h)}[v]
        return affine(img, msk, np.float32([[1, 0, dx], [0, 1, dy]]))
    if p == 'scale':
        return affine(img, msk, cv2.getRotationMatrix2D(c, 0, 1 + l if v == 'in' else 1 - l))
    if p == 'rotate':
        return affine(img, msk, cv2.getRotationMatrix2D(c, l if v == '+' else -l, 1.0))
    x = img.astype(np.float32)
    if p == 'brightness':
        x = x + (l if v == '+' else -l)
    elif p == 'contrast':
        x = (x - x.mean()) * (1 + l if v == '+' else 1 - l) + x.mean()
    elif p == 'gamma':
        x = 255.0 * (x / 255.0) ** (1 + l if v == '+' else 1 / (1 + l))
    elif p == 'blur':
        x = cv2.GaussianBlur(x, (0, 0), l)
    elif p == 'speckle':
        rng = np.random.default_rng(zlib.crc32(f'{case}|{l}'.encode()))     # 固定種子：每次執行相同
        x = x * (1 + l * rng.standard_normal((h, w, 1)).astype(np.float32))  # 三個通道同一雜訊（灰階影像）
    return np.clip(np.rint(x), 0, 255).astype(np.uint8), msk


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def load_model(name):
    D = f'{args.output_dir}/{name}'
    config = yaml.load(open(f'{D}/config.yml'), Loader=yaml.FullLoader)
    ck = f'{D}/model.pth' if args.which == 'best' else f'{D}/model_last.pth'
    os.path.exists(ck) or sys.exit(f'找不到 {ck}')
    m = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'],
                                       embed_dims=config['input_list'], no_kan=config.get('no_kan', False))
    m.load_state_dict(torch.load(ck, map_location='cpu'), strict=True)
    print(f'模型：{ck}（move_to_train = {config.get("move_to_train")}）')
    return m.to(args.device).eval(), config


mv = list(pd.read_csv(args.list, encoding='utf-8-sig').case.astype(str))
models = {n: load_model(n) for n in args.models.split(',')}
config = next(iter(models.values()))[1]
H, W = config['input_h'], config['input_w']
base = os.path.join(config['data_dir'], config['dataset'])
tf = Compose([Resize(H, W), transforms.Normalize()])            # 與 predict_busi.py / train.py 的 val_transform 相同

cases = [('moved', i) for i in mv]
if args.ref > 0:                                                 # 原本就在訓練集的影像（兩個模型都看過）
    ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(base, 'images', '*.png')))
    tr, _ = train_test_split(ids, test_size=0.2, random_state=config['dataseed'])
    rng = np.random.default_rng(0)
    cases += [('ref_train', i) for i in rng.choice(sorted(set(tr) - set(mv)), args.ref, replace=False)]


@torch.no_grad()
def predict(model, img):
    x = tf(image=img)['image'].astype('float32') / 255
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(args.device)
    return (torch.sigmoid(model(x))[0, 0] >= 0.5).cpu().numpy()


os.makedirs(f'{args.out}/png', exist_ok=True)
rows, shows = [], {}
for k, (grp, i) in enumerate(cases):
    img = cv2.imread(os.path.join(base, 'images', i + '.png'))
    gt = cv2.imread(os.path.join(base, 'masks', '0', i + '_mask.png'), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)   # 與 albumentations Resize 相同（之後 256→256 不變）
    g = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST) > 127
    for p, s, l, v in CONDS:
        pi, pg = perturb(img, g, p, l, v, i)
        for name, (model, cfg) in models.items():
            pr = predict(model, pi)
            rows.append(dict(model=name, checkpoint=args.which, group=grp, case=i, perturb=p, strength=s, level=l,
                             variant=v, dice=dice(pr, pg), pred_pixels=int(pr.sum()), gt_pixels=int(pg.sum())))
            if grp == 'moved' and (p, s, v) in PNG_CONDS:
                shows[(i, name, p, s, v)] = (pi, pg, pr)
    print(f'  {k + 1}/{len(cases)} {i}', flush=True)

d = pd.DataFrame(rows)
d.to_csv(f'{args.out}/perturb_percase.csv', index=False, float_format='%.4f')

# ---------- 流程檢查：none 必須等於 percase_<which>.csv ----------
for name in models:
    f = f'{args.output_dir}/{name}/percase_{args.which}.csv'
    if os.path.exists(f):
        ref = pd.read_csv(f).set_index('case').dice
        x = d[(d.model == name) & (d.perturb == 'none')].set_index('case').dice
        print(f'檢查 {name}：none 與 {f} 最大差異 {np.abs(x - ref.reindex(x.index)).max():.2e}（應為 0）')

# ---------- 摘要 ----------
S = (d.groupby(['model', 'group', 'perturb', 'strength', 'level'], sort=False)
       .agg(n=('dice', 'size'), dice_mean=('dice', 'mean'), dice_median=('dice', 'median'), dice_min=('dice', 'min'),
            n_below_05=('dice', lambda x: int((x < 0.5).sum())))
       .reset_index())
S.to_csv(f'{args.out}/perturb_summary.csv', index=False, float_format='%.4f')
for grp in S.group.unique():
    t = S[S.group == grp].pivot_table(index=['perturb', 'strength', 'level'], columns='model', values='dice_mean', sort=False)
    nb = S[S.group == grp].pivot_table(index=['perturb', 'strength', 'level'], columns='model', values='n_below_05', sort=False).astype(int)
    t = t.join(nb.add_prefix('n<0.5 '))
    print(f'\n===== {grp}（{(d.group == grp).sum() // len(models) // len(CONDS)} 張）：平均 Dice，方向 / 正負合併 =====')
    print(t.to_string(float_format='%.3f'))

# ---------- 圖：每張影像一張，列 = 模型，欄 = 代表性變化 ----------
def draw(pi, pg, pr):
    o = pi.copy()
    for m, col in [(pg, (0, 255, 0)), (pr, (0, 0, 255))]:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cs, -1, col, 2)
    cv2.putText(o, f'{dice(pr, pg):.2f}', (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return o


for i in mv:
    grid = [np.hstack([draw(*shows[(i, name, p, s, v)]) for p, s, v in PNG_CONDS]) for name in models]
    cv2.imwrite(f'{args.out}/png/{i}.png', np.vstack(grid))
print(f'\n欄：' + '、'.join(p + (f'{s}' if s else '') + (f'-{v}' if v else '') for p, s, v in PNG_CONDS) + f'；列：{"、".join(models)}')
print(f'→ {args.out}/perturb_percase.csv、perturb_summary.csv、png/')