"""
BUSI 討論實驗彙整：所有模型的逐張結果（last）→ busi/summary/
  models.csv       每個模型一列：訓練張數、資料增強、Dropout、測試集各組、訓練集、差距
  comparisons.csv  成對比較（Wilcoxon，逐張配對）
  nine_cases.csv   9 張困難影像在每個模型的 Dice
  shift_sweep.png  整數像素平移掃描曲線
資料來源（只讀 GitHub 上的檔案）：
  對照組、Ψ 專家、moved ：psi-weighted-busi/results/percase/
  其餘實驗             ：psi-discussion/busi/<日期>_<主題>/percase/
組別（以原始官方切分定義）：test_all 130 張、test_other 115、test_low_dice 15（baseline seed2981 best 的 Dice < 0.5）、
  nine 9 張（測試集 Dice < 0.2，moved 實驗搬進訓練集的那 9 張）、train_all 517 張（原訓練集）
  搬移模型（moved、shift_moved）的測試集只剩 121 張：test_all 為那 121 張
用法：python make_summary.py --busi_repo <psi-weighted-busi> --disc <psi-discussion>
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--busi_repo', default='../../../psi-weighted-busi')
ap.add_argument('--disc', default='../..')
args = ap.parse_args()
OUT = os.path.join(args.disc, 'busi', 'summary')
os.makedirs(OUT, exist_ok=True)

# 模型：名稱 → (訓練張數, 資料增強, Dropout, 說明)
SH = 'flip+rot90 + shift ±32px'
MODELS = [
    ('base2', 517, 'flip+rot90（官方）', 0, '對照組'),
    ('base3', 517, 'flip+rot90（官方）', 0, '對照組重訓（雜訊）'),
    ('psi8w3n', 517, 'flip+rot90（官方）', 0, 'Ψ8 加權 ×3'),
    ('moved', 526, 'flip+rot90（官方）', 0, '9 張搬進訓練集'),
    ('shift', 517, SH, 0, '平移增強'),
    ('shift_r2', 517, SH, 0, '平移增強重訓'),
    ('ssr', 517, SH + ' + scale ±10% + rotate ±15°', 0, ''),
    ('ssr_int', 517, SH + ' + scale ±10% + rotate ±15° + brightness/contrast ±20% + gamma', 0, ''),
    ('psi8w3n_shift', 517, SH, 0, 'Ψ8 加權 ×3'),
    ('shift_moved', 526, SH, 0, '9 張搬進訓練集'),
    ('drop1', 517, 'flip+rot90（官方）', 0.1, ''),
    ('drop3', 517, 'flip+rot90（官方）', 0.3, ''),
    ('shift_drop1', 517, SH, 0.1, ''),
    ('shift_drop3', 517, SH, 0.3, ''),
]


def find(name, which='last'):
    f = f'busi_UKAN_{name}_{which}.csv'
    c = [os.path.join(args.busi_repo, 'results', 'percase', f)] + sorted(glob.glob(os.path.join(args.disc, 'busi', '*', 'percase', f)))
    c = [p for p in c if os.path.exists(p)]
    return (pd.read_csv(c[0]).set_index('case'), c[0]) if c else (None, None)


base, _ = find('seed2981', 'best')
nine = pd.Index(pd.read_csv(os.path.join(args.busi_repo, 'weight_lists', 'moved_busi_dice02.csv'), encoding='utf-8-sig').case.astype(str))
tr, te = base.index[base.split == 'train'], base.index[base.split == 'val']
low = base.index[(base.split == 'val') & (base.dice < 0.5)]
G = {'test_all': te, 'test_other': te.difference(low), 'test_low_dice': low, 'nine': nine, 'train_all': tr}

M, rows = {}, []
for name, ntr, aug, drop, note in MODELS:
    d, src = find(name)
    if d is None:
        continue
    M[name] = d
    moved = ntr == 526
    t = te.difference(nine) if moved else te
    r = dict(model=f'busi_UKAN_{name}', train_images=ntr, augmentation=aug, dropout=drop, note=note, n_test=len(t),
             test_all=d.dice.reindex(t).mean(), test_other=d.dice.reindex(t.difference(low)).mean(),
             test_low_dice=d.dice.reindex(t.intersection(low)).mean(), nine=d.dice.reindex(nine).mean(),
             nine_seen=moved, train_all=d.dice.reindex(tr).mean(), source=os.path.relpath(src, args.disc))
    r['gap'] = r['train_all'] - r['test_all']
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/models.csv', index=False, float_format='%.4f', encoding='utf-8-sig')

COMPS = [('noise_noaug', 'base2', 'base3'), ('noise_shift', 'shift', 'shift_r2'),
         ('shift', 'base2', 'shift'), ('shift_r2', 'base2', 'shift_r2'), ('shift_vs_base3', 'base3', 'shift'), ('shift_r2_vs_base3', 'base3', 'shift_r2'),
         ('ssr', 'shift', 'ssr'), ('ssr_int', 'ssr', 'ssr_int'),
         ('psi8_noaug', 'base2', 'psi8w3n'), ('psi8_noaug_vs_base3', 'base3', 'psi8w3n'),
         ('psi8_shift', 'shift', 'psi8w3n_shift'), ('psi8_shift_r2', 'shift_r2', 'psi8w3n_shift'),
         ('drop1', 'base2', 'drop1'), ('drop1_vs_base3', 'base3', 'drop1'), ('drop3', 'base2', 'drop3'), ('drop3_vs_base3', 'base3', 'drop3'),
         ('shift_drop1', 'shift', 'shift_drop1'), ('shift_drop1_vs_r2', 'shift_r2', 'shift_drop1'),
         ('shift_drop3', 'shift', 'shift_drop3'), ('shift_drop3_vs_r2', 'shift_r2', 'shift_drop3')]
rows = []
for comp, a, b in COMPS:
    if a not in M or b not in M:
        continue
    for g in ['test_all', 'test_other', 'test_low_dice', 'nine', 'train_all']:
        x, y = M[a].dice.reindex(G[g]), M[b].dice.reindex(G[g])
        dd = (y - x).to_numpy()
        p = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else np.nan
        rows.append(dict(comparison=comp, model_a=f'busi_UKAN_{a}', model_b=f'busi_UKAN_{b}', group=g, n=len(dd), dice_a=x.mean(),
                         dice_b=y.mean(), delta=dd.mean(), n_up=int((dd > 0).sum()), n_down=int((dd < 0).sum()), wilcoxon_p=p))
C = pd.DataFrame(rows)
C['wilcoxon_p'] = C.wilcoxon_p.map(lambda v: '' if pd.isna(v) else (f'{v:.1e}' if v < 0.001 else f'{v:.4f}'))
C.to_csv(f'{OUT}/comparisons.csv', index=False, float_format='%.4f')

N = pd.DataFrame({'baseline_best': base.dice.reindex(nine)})
for k, d in M.items():
    N[k] = d.dice.reindex(nine)
N.index.name = 'case'
N.to_csv(f'{OUT}/nine_cases.csv', float_format='%.4f')

# 平移掃描曲線
S = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.disc, 'busi', '*', 'perturb_sweep', 'perturb_summary.csv')))
               + sorted(glob.glob(os.path.join(args.disc, 'busi', '*_perturb_sweep', 'perturb_summary.csv')))])
S = S.drop_duplicates(['model', 'group', 'level'])
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'Microsoft JhengHei', 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=150)
    for m, g, lab, c, ls in [('busi_UKAN_moved', 'ref_train', 'normal 30 train images, moved (no shift aug)', '#9a9a9a', '-'),
                             ('busi_UKAN_shift', 'ref_train', 'normal 30 train images, shift (shift aug)', '#9a9a9a', '--'),
                             ('busi_UKAN_moved', 'moved', 'the 9 hard images, moved (seen, no shift aug)', '#1f6fd1', '-'),
                             ('busi_UKAN_shift_moved', 'moved', 'the 9 hard images, shift_moved (seen, shift aug)', '#1f6fd1', '--')]:
        s = S[(S.model == m) & (S.group == g)].sort_values('level')
        if len(s):
            ax.plot(s.level, s.dice_mean, ls, color=c, marker='o', ms=3, lw=2, label=lab)
    for v in (32, 64):
        ax.axvline(v, color='#bbbbbb', ls=':', lw=1)
    ax.text(33, 0.04, 'shift = 32, 64 px (multiples of the 32x downsampling)', fontsize=8, color='#666666')
    ax.set_xlabel('integer shift (px, mean of right and down)')
    ax.set_ylabel('mean Dice')
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48, 64])
    ax.grid(axis='y', color='#eeeeee')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    ax.set_title('Dice drops whenever the shift is not a multiple of 32 px; shift augmentation flattens it', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUT}/shift_sweep.png')
    print(f'→ {OUT}/shift_sweep.png')
except ImportError:
    print('沒有 matplotlib，略過 shift_sweep.png')

pd.set_option('display.width', 250)
print(T.drop(columns=['augmentation', 'source']).to_string(index=False, float_format='%.4f'))
print(f'→ {OUT}/models.csv、comparisons.csv、nine_cases.csv')
