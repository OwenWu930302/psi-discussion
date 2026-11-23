"""
分組搬移：那 9 張分成 3 組（每組 3 張），每次把另外 6 張搬進訓練集，看沒搬的 3 張（held_out）會不會變好
  比較：base2（9 張都沒看過）、moved（9 張都看過）、busi_UKAN_moved_g<k>（看過另外 6 張，沒看過這 3 張）；全部 last
輸出（outputs/analysis_busi_group/）
  group_BUSI_percase.csv  每個分組模型 × 那 9 張一列：held_out（這張是否沒搬進去）、base2、moved、該分組模型、與 base2 的差
  group_BUSI_summary.csv  每個分組模型一列：held_out 3 張、搬進去的 6 張、其餘測試影像 121 張的平均
用法（Seg_UKAN 資料夾）：python analyze_group.py --groups <分組清單資料夾>
"""
import argparse
import glob
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--groups', default='outputs/group_lists')
ap.add_argument('--list', default='/mnt/data2t/BUSI/psi_lists/moved_busi_dice02.csv')
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--out', default='outputs/analysis_busi_group')
args = ap.parse_args()
L = lambda n: pd.read_csv(f'{args.output_dir}/{n}/percase_last.csv').set_index('case').dice
nine = list(pd.read_csv(args.list, encoding='utf-8-sig').case.astype(str))
b2, mv = L('busi_UKAN_base2'), L('busi_UKAN_moved')
split = pd.read_csv(f'{args.output_dir}/busi_UKAN_base2/percase_last.csv').set_index('case').split
rest = [c for c in split.index[split == 'val'] if c not in nine]
P, S = [], []
for f in sorted(glob.glob(f'{args.groups}/moved_g*.csv')):
    g = os.path.basename(f)[len('moved_'):-4]
    name = f'busi_UKAN_moved_{g}'
    if not os.path.exists(f'{args.output_dir}/{name}/percase_last.csv'):
        print(f'略過（尚未完成）：{name}')
        continue
    d = L(name)
    moved_in = set(pd.read_csv(f, encoding='utf-8-sig').case.astype(str))
    held = [c for c in nine if c not in moved_in]
    for c in nine:
        P.append(dict(group=g, model=name, case=c, held_out='yes' if c in held else 'no', base2=b2[c], moved=mv[c],
                      group_model=d[c], delta_vs_base2=d[c] - b2[c]))
    seen = [c for c in nine if c in moved_in]
    S.append(dict(group=g, model=name, held_out_cases='; '.join(held), n_held_out=len(held),
                  held_out_base2=b2[held].mean(), held_out_group_model=d[held].mean(), held_out_moved=mv[held].mean(),
                  n_held_out_above_05=int((d[held] >= 0.5).sum()), seen_6_group_model=d[seen].mean(),
                  remaining_121_base2=b2[rest].mean(), remaining_121_group_model=d[rest].mean()))
os.makedirs(args.out, exist_ok=True)
P, S = pd.DataFrame(P), pd.DataFrame(S)
P.to_csv(f'{args.out}/group_BUSI_percase.csv', index=False, float_format='%.4f', encoding='utf-8-sig')
S.to_csv(f'{args.out}/group_BUSI_summary.csv', index=False, float_format='%.4f', encoding='utf-8-sig')
pd.set_option('display.width', 250)
if len(P):
    print(P[P.held_out == 'yes'].to_string(index=False, float_format='%.3f'))
    print(S.drop(columns=['held_out_cases']).to_string(index=False, float_format='%.3f'))
print(f'→ {args.out}/group_BUSI_percase.csv、group_BUSI_summary.csv')