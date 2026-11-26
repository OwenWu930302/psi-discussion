"""
BUSI 討論實驗的分析：逐張結果（last）→ busi/results/analysis/（格式比照 psi-weighted-ukan/results/analysis）
輸入
  對照組、Ψ8 專家、moved、baseline：psi-weighted-ukan/results/percase/
  本倉庫的模型                     ：busi/results/percase/busi_UKAN_<模型>_last.csv
  擾動測試逐張                     ：busi/results/analysis/perturb_BUSI_percase.csv、sweep_BUSI_percase.csv
輸出
  aug_BUSI_models.csv    每個模型一列（依類別排序）：類別、訓練資料夾名稱、訓練張數、資料增強、Dropout、Ψ 加權、測試集各組、訓練集、差距
  aug_BUSI_summary.csv   成對比較（control → model），各組的前後變化與 Wilcoxon
  aug_BUSI_percase.csv   成對比較的逐張前後（測試集 130 張；搬移比較另含那 9 張）
  low_dice_BUSI_percase.csv、low_dice_BUSI_wide.csv、low_dice_BUSI_summary.csv  低分 15 張每個模型逐張的前後
  low_other_BUSI_percase.csv、_wide.csv、_summary.csv、_case_summary.csv  其中不在那 9 張的 6 張，單獨整理
  perturb_BUSI_summary.csv、sweep_BUSI_summary.csv、shift_sweep_BUSI.png
模型類別（category，所有表都依此順序）：
  control 對照組 → moved 9 張搬進訓練集 → augmentation 資料增強 → regularization 通用正則化（Dropout、雜訊）
  → combination 平移增強 + Dropout → psi_weighting Ψ 加權
模型名稱：表中用 drop10 / drop30（Dropout 10% / 30%）；訓練資料夾與逐張檔名仍是 drop1 / drop3（train_name 欄）
組別（以官方原始切分定義）：all 130、other 115、low_dice 15（baseline seed2981 best 的 Dice < 0.5）、
  nine 9（moved_busi_dice02.csv）、train 517（原訓練集）；搬移模型的測試集只剩 121 張（remaining）
用法（psi-discussion 根目錄，兩個倉庫 clone 在同一層）：
  python busi/code/make_busi_results.py --busi_repo ../psi-weighted-ukan --disc .
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--busi_repo', default='../psi-weighted-ukan')
ap.add_argument('--disc', default='.')
args = ap.parse_args()
PC = os.path.join(args.disc, 'busi', 'results', 'percase')
OUT = os.path.join(args.disc, 'busi', 'results', 'analysis')

# 模型：(類別, 訓練名稱, 訓練張數, 資料增強, Dropout, Ψ 加權)；順序 = 所有輸出表的順序
OFF = 'flip+rot90'
SH = OFF + ' + shift32'
SSR = SH + ' + scale10 + rotate15'
MODELS = [('control', 'base2', 517, OFF, 0, 'none'), ('control', 'base3', 517, OFF, 0, 'none'),
          ('moved', 'moved', 526, OFF, 0, 'none'), ('moved', 'shift_moved', 526, SH, 0, 'none'),
          ('augmentation', 'shift', 517, SH, 0, 'none'), ('augmentation', 'shift_r2', 517, SH, 0, 'none'),
          ('augmentation', 'ssr', 517, SSR, 0, 'none'), ('augmentation', 'ssr_int', 517, SSR + ' + intensity20', 0, 'none'),
          ('regularization', 'drop1', 517, OFF, 0.1, 'none'), ('regularization', 'drop3', 517, OFF, 0.3, 'none'),
          ('regularization', 'drop3_r2', 517, OFF, 0.3, 'none'), ('regularization', 'speckle', 517, OFF + ' + speckle10', 0, 'none'),
          ('combination', 'shift_drop1', 517, SH, 0.1, 'none'), ('combination', 'shift_drop3', 517, SH, 0.3, 'none'),
          ('psi_weighting', 'psi8w3n', 517, OFF, 0, 'psi8 x3'), ('psi_weighting', 'psi8w3n_shift', 517, SH, 0, 'psi8 x3')]
LABEL = {'drop1': 'drop10', 'drop3': 'drop30', 'drop3_r2': 'drop30_r2', 'shift_drop1': 'shift_drop10', 'shift_drop3': 'shift_drop30'}
CAT = {k: c for c, k, *_ in MODELS}
short = lambda k: LABEL.get(k, k)          # 表中顯示的名稱
D = lambda k: f'busi_UKAN_{short(k)}'
COMPS = [('noise_noaug', 'base2', 'base3'),
         ('moved_noaug', 'base2', 'moved'), ('moved_shift', 'shift', 'shift_moved'), ('moved_shift', 'moved', 'shift_moved'),
         ('noise_shift', 'shift', 'shift_r2'),
         ('shift', 'base2', 'shift'), ('shift', 'base3', 'shift'), ('shift_r2', 'base2', 'shift_r2'), ('shift_r2', 'base3', 'shift_r2'),
         ('ssr', 'shift', 'ssr'), ('ssr', 'shift_r2', 'ssr'), ('ssr_int', 'ssr', 'ssr_int'),
         ('drop10', 'base2', 'drop1'), ('drop10', 'base3', 'drop1'), ('drop30', 'base2', 'drop3'), ('drop30', 'base3', 'drop3'),
         ('noise_drop30', 'drop3', 'drop3_r2'), ('drop30_r2', 'base2', 'drop3_r2'), ('drop30_r2', 'base3', 'drop3_r2'),
         ('speckle', 'base2', 'speckle'), ('speckle', 'base3', 'speckle'),
         ('shift_drop10', 'shift', 'shift_drop1'), ('shift_drop10', 'shift_r2', 'shift_drop1'),
         ('shift_drop30', 'shift', 'shift_drop3'), ('shift_drop30', 'shift_r2', 'shift_drop3'),
         ('psi8_noaug', 'base2', 'psi8w3n'), ('psi8_noaug', 'base3', 'psi8w3n'),
         ('psi8_shift', 'shift', 'psi8w3n_shift'), ('psi8_shift', 'shift_r2', 'psi8w3n_shift')]


def load(name, which='last'):
    for d in [os.path.join(args.busi_repo, 'results', 'percase'), PC]:
        f = os.path.join(d, f'busi_UKAN_{name}_{which}.csv')
        if os.path.exists(f):
            return pd.read_csv(f).set_index('case').dice
    return None


def pval(x):
    return float(wilcoxon(x).pvalue) if len(x) > 1 and np.any(x != 0) else np.nan


R = lambda f: pd.read_csv(f, encoding='utf-8-sig')
base = R(os.path.join(args.busi_repo, 'results', 'percase', 'busi_UKAN_seed2981_best.csv')).set_index('case')
nine = pd.Index(R(os.path.join(args.busi_repo, 'weight_lists', 'moved_busi_dice02.csv')).case.astype(str))
tr, te = base.index[base.split == 'train'], base.index[base.split == 'val']
low = base.index[(base.split == 'val') & (base.dice < 0.5)]
M = {k: v for k, v in ((k, load(k)) for _, k, *_ in MODELS) if v is not None}
print('模型：' + '、'.join(M))

rows = []
for cat, name, ntr, aug, drop, psi in MODELS:
    if name not in M:
        continue
    d, t = M[name], (te.difference(nine) if ntr == 526 else te)
    r = dict(category=cat, model=D(name), train_name=f'busi_UKAN_{name}', train_images=ntr, augmentation=aug, dropout=drop, psi_weight=psi, n_test=len(t),
             test_all=d.reindex(t).mean(), test_other=d.reindex(t.difference(low)).mean(),
             test_low_dice=d.reindex(t.intersection(low)).mean(), nine=d.reindex(nine).mean(),
             nine_in_train='yes' if ntr == 526 else 'no', train_all=d.reindex(tr).mean())
    r['gap'] = r['train_all'] - r['test_all']
    rows.append(r)
os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_csv(f'{OUT}/aug_BUSI_models.csv', index=False, float_format='%.4f', encoding='utf-8-sig')

S, P = [], []
for comp, a, b in COMPS:
    if a not in M or b not in M:
        continue
    moved = 'moved' in comp
    groups = ([('nine', nine), ('remaining', te.difference(nine)), ('train', tr)] if moved else
              [('all', te), ('other', te.difference(low)), ('low_dice', low), ('nine', nine), ('train', tr)])
    for g, idx in groups:
        x, y = M[a].reindex(idx), M[b].reindex(idx)
        dd = (y - x).to_numpy()
        S.append(dict(category=CAT[b], comparison=comp, control=D(a), model=D(b), group=g, n=len(idx), before_control=x.mean(),
                      after_model=y.mean(), delta=dd.mean(), n_up=int((dd > 0).sum()), n_down=int((dd < 0).sum()), wilcoxon_p=pval(dd)))
    idx = te.union(nine)
    for c in idx:
        P.append(dict(comparison=comp, control=D(a), model=D(b), case=c,
                      split='test' if (c in te and not (moved and c in nine)) else 'train',
                      low_dice='yes' if c in low else 'no', nine='yes' if c in nine else 'no',
                      before_control=M[a].get(c), after_model=M[b].get(c), delta=M[b].get(c) - M[a].get(c)))
S = pd.DataFrame(S)
S['wilcoxon_p'] = S.wilcoxon_p.map(lambda v: '' if pd.isna(v) else (f'{v:.1e}' if v < 0.001 else f'{v:.4f}'))
S.to_csv(f'{OUT}/aug_BUSI_summary.csv', index=False, float_format='%.4f', encoding='utf-8-sig')
pd.DataFrame(P).to_csv(f'{OUT}/aug_BUSI_percase.csv', index=False, float_format='%.4f', encoding='utf-8-sig')

# 低分 15 張：每個模型逐張的前後（before = 對照組 base2，after = 該模型；last）
#   low_dice_BUSI_percase.csv  一個模型 × 一張影像一列（格式同 psi-weighted-ukan 的 test_BUSI_percase.csv）
#   low_dice_BUSI_wide.csv     一張影像一列、每個模型一欄（方便直接看）
#   low_dice_BUSI_summary.csv  每個模型在 low_dice（15）、nine（9）、low_other（6）三組的平均前後
ctrl = M['base2']
P, W = [], pd.DataFrame(index=low)
W['nine'] = ['yes' if c in nine else 'no' for c in low]
W['baseline_dice'] = base.dice.reindex(low)
for _, name, ntr, *_ in MODELS:
    if name not in M:
        continue
    W[short(name)] = M[name].reindex(low)
    if name == 'base2':
        continue
    for c in low:
        P.append(dict(category=CAT[name], model=D(name), case=c, split='train' if (ntr == 526 and c in nine) else 'test',
                      nine='yes' if c in nine else 'no', baseline_dice=base.dice.get(c), before_control=ctrl.get(c),
                      after_model=M[name].get(c), delta=M[name].get(c) - ctrl.get(c)))
P = pd.DataFrame(P)
P.to_csv(f'{OUT}/low_dice_BUSI_percase.csv', index=False, float_format='%.4f', encoding='utf-8-sig')
W = W.sort_values(['nine', 'baseline_dice'], ascending=[False, True])
W.index.name = 'case'
W.to_csv(f'{OUT}/low_dice_BUSI_wide.csv', float_format='%.4f', encoding='utf-8-sig')
L = []
for (cat, m), g in P.groupby(['category', 'model'], sort=False):
    for grp, sel in [('low_dice', g), ('nine', g[g.nine == 'yes']), ('low_other', g[g.nine == 'no'])]:
        dd = (sel.after_model - sel.before_control).to_numpy()
        L.append(dict(category=cat, model=m, control='busi_UKAN_base2', group=grp, n=len(sel), before_control=sel.before_control.mean(),
                      after_model=sel.after_model.mean(), delta=dd.mean(), n_up=int((dd > 0).sum()), n_down=int((dd < 0).sum()),
                      n_above_05=int((sel.after_model >= 0.5).sum()), wilcoxon_p=pval(dd)))
L = pd.DataFrame(L)
L['wilcoxon_p'] = L.wilcoxon_p.map(lambda v: '' if pd.isna(v) else (f'{v:.1e}' if v < 0.001 else f'{v:.4f}'))
L.to_csv(f'{OUT}/low_dice_BUSI_summary.csv', index=False, float_format='%.4f', encoding='utf-8-sig')

# 低分但不在那 9 張的 6 張（low_other）：單獨整理
#   low_other_BUSI_percase.csv       一個模型 × 一張影像一列（before = base2，after = 該模型）
#   low_other_BUSI_wide.csv          一張影像一列、每個模型一欄
#   low_other_BUSI_summary.csv       每個模型一列：6 張的平均前後、升降、Dice ≥ 0.5 張數、Wilcoxon
#   low_other_BUSI_case_summary.csv  每張影像一列：baseline（best）、base2、base3（last）、其他模型的平均 / 最低 / 最高、升降模型數
O = P[P.nine == 'no'].drop(columns=['nine', 'category'])
O.to_csv(f'{OUT}/low_other_BUSI_percase.csv', index=False, float_format='%.4f', encoding='utf-8-sig')
W[W.nine == 'no'].drop(columns=['nine']).to_csv(f'{OUT}/low_other_BUSI_wide.csv', float_format='%.4f', encoding='utf-8-sig')
L[L.group == 'low_other'].drop(columns=['group']).to_csv(f'{OUT}/low_other_BUSI_summary.csv', index=False, float_format='%.4f', encoding='utf-8-sig')
C = []
for c, g in O.groupby('case', sort=False):
    C.append(dict(case=c, baseline_best=g.baseline_dice.iloc[0], base2_last=g.before_control.iloc[0], base3_last=M['base3'].get(c) if 'base3' in M else np.nan,
                  n_models=len(g), after_mean=g.after_model.mean(), after_min=g.after_model.min(), after_max=g.after_model.max(),
                  after_range=g.after_model.max() - g.after_model.min(), delta_mean=g.delta.mean(),
                  n_up=int((g.delta > 0).sum()), n_down=int((g.delta < 0).sum()), n_above_05=int((g.after_model >= 0.5).sum())))
pd.DataFrame(C).sort_values('delta_mean', ascending=False).to_csv(f'{OUT}/low_other_BUSI_case_summary.csv', index=False,
                                                                  float_format='%.4f', encoding='utf-8-sig')

# 擾動測試摘要（方向 / 正負合併）
for kind, extra in [('perturb', {}), ('sweep', {'prob_gt_mean': ('prob_gt', 'mean')})]:
    d = R(f'{OUT}/{kind}_BUSI_percase.csv')
    d['model'] = d.model.map(lambda m: D(m[len('busi_UKAN_'):]))   # drop1 → drop10 等
    s = (d.groupby(['model', 'group', 'perturb', 'strength', 'level'], sort=False)
          .agg(n=('dice', 'size'), dice_mean=('dice', 'mean'), n_below_05=('dice', lambda v: int((v < 0.5).sum())), **extra).reset_index())
    order = {D(k): i for i, (_, k, *_) in enumerate(MODELS)}
    s = s.assign(_o=s.model.map(order)).sort_values('_o', kind='stable').drop(columns='_o')
    s.to_csv(f'{OUT}/{kind}_BUSI_summary.csv', index=False, float_format='%.4f', encoding='utf-8-sig')

# 平移掃描曲線
s = R(f'{OUT}/sweep_BUSI_summary.csv')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=150)
    for m, g, lab, c, ls in [('busi_UKAN_moved', 'train_ref30', 'normal 30 train images, moved (no shift aug)', '#9a9a9a', '-'),
                             ('busi_UKAN_shift', 'train_ref30', 'normal 30 train images, shift (shift aug)', '#9a9a9a', '--'),
                             ('busi_UKAN_moved', 'nine', 'the 9 hard images, moved (seen, no shift aug)', '#1f6fd1', '-'),
                             ('busi_UKAN_shift_moved', 'nine', 'the 9 hard images, shift_moved (seen, shift aug)', '#1f6fd1', '--')]:
        t = s[(s.model == m) & (s.group == g)].sort_values('level')
        if len(t):
            ax.plot(t.level, t.dice_mean, ls, color=c, marker='o', ms=3, lw=2, label=lab)
    for v in (32, 64):
        ax.axvline(v, color='#bbbbbb', ls=':', lw=1)
    ax.text(33, 0.04, 'shift = 32, 64 px (multiples of the 32x downsampling)', fontsize=8, color='#666666')
    ax.set(xlabel='integer shift (px, mean of right and down)', ylabel='mean Dice', ylim=(0, 1.02), xticks=[0, 8, 16, 24, 32, 40, 48, 64])
    ax.grid(axis='y', color='#eeeeee')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    ax.set_title('Dice drops whenever the shift is not a multiple of 32 px; shift augmentation flattens it', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUT}/shift_sweep_BUSI.png')
except ImportError:
    print('沒有 matplotlib，略過 shift_sweep_BUSI.png')
print(f'→ {OUT}/')