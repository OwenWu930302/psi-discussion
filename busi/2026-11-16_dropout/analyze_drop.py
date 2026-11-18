"""
BUSI Dropout 實驗的分析（全部用 last；低分組 low_dice = baseline busi_UKAN_seed2981 best 測試集 Dice < 0.5）
  drop_models.csv  ：每個模型一列（測試集各組、訓練集、訓練−測試差距）
  drop_summary.csv ：成對比較（model_a → model_b）
    noise_noaug  base2 → base3              noise_shift   shift → shift_r2         （雜訊基準）
    drop1        base2 → drop1              drop1_b3      base3 → drop1
    drop3        base2 → drop3              drop3_b3      base3 → drop3
    shift_drop1  shift → shift_drop1        shift_drop1_r2  shift_r2 → shift_drop1
    shift_drop3  shift → shift_drop3        shift_drop3_r2  shift_r2 → shift_drop3
  組別：test_all / test_nine / test_low_dice / test_other / train_all
用法（Seg_UKAN 資料夾）：python analyze_drop.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--list', default='/mnt/data2t/BUSI/psi_lists/moved_busi_dice02.csv')
ap.add_argument('--out', default='outputs/analysis_busi_drop')
args = ap.parse_args()


def load(name, which='last'):
    f = f'{args.output_dir}/{name}/percase_{which}.csv'
    return pd.read_csv(f).set_index('case') if os.path.exists(f) else None


def wil(x):
    x = np.asarray(x, float)
    return float(wilcoxon(x).pvalue) if len(x) > 1 and np.any(x != 0) else np.nan


nine = pd.Index(pd.read_csv(args.list, encoding='utf-8-sig').case.astype(str))
base = load('busi_UKAN_seed2981', 'best')
tr, te = base.index[base.split == 'train'], base.index[base.split == 'val']
low = base.index[(base.split == 'val') & (base.dice < 0.5)]
G = [('test_all', te), ('test_nine', nine), ('test_low_dice', low), ('test_other', te.difference(low)), ('train_all', tr)]
NAMES = ['base2', 'base3', 'drop1', 'drop3', 'shift', 'shift_r2', 'shift_drop1', 'shift_drop3']
M = {k: load(f'busi_UKAN_{k}') for k in NAMES}
M = {k: v for k, v in M.items() if v is not None}
print('模型：' + '、'.join(M))

rows = []
for k, v in M.items():
    r = dict(model=f'busi_UKAN_{k}', **{g: v.dice.reindex(i).mean() for g, i in G})
    r['gap_train_test'] = r['train_all'] - r['test_all']
    rows.append(r)
T = pd.DataFrame(rows)

COMPS = [('noise_noaug', 'base2', 'base3'), ('noise_shift', 'shift', 'shift_r2'),
         ('drop1', 'base2', 'drop1'), ('drop1_b3', 'base3', 'drop1'), ('drop3', 'base2', 'drop3'), ('drop3_b3', 'base3', 'drop3'),
         ('shift_drop1', 'shift', 'shift_drop1'), ('shift_drop1_r2', 'shift_r2', 'shift_drop1'),
         ('shift_drop3', 'shift', 'shift_drop3'), ('shift_drop3_r2', 'shift_r2', 'shift_drop3')]
rows = []
for comp, a, b in COMPS:
    if a not in M or b not in M:
        continue
    for g, idx in G:
        x, y = M[a].dice.reindex(idx), M[b].dice.reindex(idx)
        d = y - x
        rows.append(dict(comparison=comp, model_a=f'busi_UKAN_{a}', model_b=f'busi_UKAN_{b}', group=g, n=len(idx),
                         dice_a=x.mean(), dice_b=y.mean(), delta=d.mean(), n_up=int((d > 0).sum()),
                         n_down=int((d < 0).sum()), wilcoxon_p=wil(d)))
S = pd.DataFrame(rows)
os.makedirs(args.out, exist_ok=True)
T.to_csv(f'{args.out}/drop_models.csv', index=False, float_format='%.4f')
S.to_csv(f'{args.out}/drop_summary.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 220)
print(T.to_string(index=False, float_format='%.4f'))
print()
print(S[S.group.isin(['test_all', 'test_nine', 'test_low_dice', 'train_all'])].to_string(index=False, float_format='%.4f'))
print(f'→ {args.out}/drop_models.csv、drop_summary.csv')