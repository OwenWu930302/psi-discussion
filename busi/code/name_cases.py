"""把 failure_diagnostics.py 的 per_case.csv（case = 0..N−1）換成影像名稱，並加上 nine（是否那 9 張）
用法：python name_cases.py <npz> <per_case.csv> <moved_busi_dice02.csv>"""
import sys

import numpy as np
import pandas as pd

cases = np.load(sys.argv[1], allow_pickle=True)['cases']
d = pd.read_csv(sys.argv[2])
nine = set(pd.read_csv(sys.argv[3], encoding='utf-8-sig').iloc[:, 0].astype(str))
d.insert(1, 'nine', ['yes' if c in nine else 'no' for c in cases[d.case]])
d['case'] = cases[d.case]
d.to_csv(sys.argv[2], index=False)
print(f'{sys.argv[2]}：{len(d)} 列，已換成影像名稱')