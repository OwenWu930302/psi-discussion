"""
BUSI 平移增強實驗的分析（全部用 last；低分組 low_dice = baseline busi_UKAN_seed2981 best 測試集 Dice < 0.5）
  noise              base2 → base3          兩次不加權訓練的差距（雜訊基準）
  shift              base2 → shift          只差平移增強；9 張在測試集（沒看過）
  moved_noaug        base2 → moved          看過 9 張（無平移增強）
  moved_shift        shift → shift_moved    看過 9 張（有平移增強）
  shift_on_moved     moved → shift_moved    兩者都看過 9 張，只差平移增強
組別：test_all / test_nine / test_low_dice / test_other（測試集）、train_all（原本的訓練集 517 張）
      搬移模型的測試集只剩 121 張：test_remaining；9 張改稱 nine（看過或沒看過依模型而定）
輸出（outputs/analysis_busi_shift/）：shift_summary.csv、shift_nine.csv
用法（Seg_UKAN 資料夾）：python analyze_shift.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--list', default='/mnt/data2t/BUSI/psi_lists/moved_busi_dice02.csv')
ap.add_argument('--out', default='outputs/analysis_busi_shift')
args = ap.parse_args()
O = args.output_dir


def load(name, which='last'):
    f = f'{O}/{name}/percase_{which}.csv'
    return pd.read_csv(f).set_index('case') if os.path.exists(f) else None


def wil(x):
    x = np.asarray(x, float)
    return float(wilcoxon(x).pvalue) if len(x) > 1 and np.any(x != 0) else np.nan


nine = list(pd.read_csv(args.list, encoding='utf-8-sig').case.astype(str))
base = load('busi_UKAN_seed2981', 'best')
orig_train = base.index[base.split == 'train']
orig_test = base.index[base.split == 'val']
low = base.index[(base.split == 'val') & (base.dice < 0.5)]
rest = orig_test.difference(nine)
M = {k: load(f'busi_UKAN_{k}') for k in ['base2', 'base3', 'shift', 'moved', 'shift_moved']}
print('模型：' + '、'.join(f'{k}{"" if v is not None else "（缺）"}' for k, v in M.items()))

GROUPS_TEST = [('test_all', orig_test), ('test_nine', pd.Index(nine)), ('test_low_dice', low),
               ('test_other', orig_test.difference(low)), ('train_all', orig_train)]
GROUPS_MOVED = [('nine', pd.Index(nine)), ('test_remaining', rest), ('train_all', orig_train)]
COMPS = [('noise', 'base2', 'base3', GROUPS_TEST), ('shift', 'base2', 'shift', GROUPS_TEST),
         ('moved_noaug', 'base2', 'moved', GROUPS_MOVED), ('moved_shift', 'shift', 'shift_moved', GROUPS_MOVED),
         ('shift_on_moved', 'moved', 'shift_moved', GROUPS_MOVED)]
rows = []
for comp, a, b, groups in COMPS:
    if M[a] is None or M[b] is None:
        continue
    for g, idx in groups:
        x, y = M[a].dice.reindex(idx), M[b].dice.reindex(idx)
        d = y - x
        rows.append(dict(comparison=comp, model_a=f'busi_UKAN_{a}', model_b=f'busi_UKAN_{b}', group=g, n=len(idx),
                         dice_a=x.mean(), dice_b=y.mean(), delta=d.mean(), n_up=int((d > 0).sum()),
                         n_down=int((d < 0).sum()), wilcoxon_p=wil(d)))
S = pd.DataFrame(rows)
os.makedirs(args.out, exist_ok=True)
S.to_csv(f'{args.out}/shift_summary.csv', index=False, float_format='%.4f')
N = pd.DataFrame({'baseline_best': base.dice.reindex(nine)})
for k, v in M.items():
    if v is not None:
        N[k] = v.dice.reindex(nine)
N.index.name = 'case'
N.to_csv(f'{args.out}/shift_nine.csv', float_format='%.4f')
pd.set_option('display.width', 200)
print(S.to_string(index=False, float_format='%.4f'))
print('\n9 張（last；baseline 為 seed2981 best）\n' + N.to_string(float_format='%.3f'))
print(f'→ {args.out}/shift_summary.csv、shift_nine.csv')