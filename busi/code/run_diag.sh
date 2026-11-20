#!/bin/bash
# 用 failure_diagnostics.py 診斷 BUSI 測試集的失敗（結果只放 psi-discussion）
#   seed2981（baseline，best；Ψ 與加權清單都由它而來）：候選排名 recall@k + 失敗類型、Ψ 的 AUROC / 門檻、梯度對齊
#   base2、shift（last）：只做候選排名 + 失敗類型（看平移增強有沒有改變失敗的類型）
#   Ψ 門檻與 psi-weighted-busi 的 test_BUSI_selection.csv 相同（failure_diagnostics 用「>」，所以 ≥ 的門檻略降 1e-7）
#   形狀特徵（只由遮罩算）：psi1_convex、psi6_cc；psi8_rotation 需要模型推論，不算形狀特徵
# 用法（Seg_UKAN 資料夾、ukan 環境）：bash run_diag.sh > run_diag.log 2>&1
cd "$(dirname "$0")"
DISC=${DISC:-$HOME/桌面/論文/psi-discussion}
MLIST=${MLIST:-/mnt/data2t/BUSI/psi_lists/moved_busi_dice02.csv}
FLIST=${FLIST:-/mnt/data2t/BUSI/psi_lists/wcases_busi_psi8_rotation.csv}
OUT=outputs/diag
TAUS="--tau psi1_convex=1.0512699 --tau psi6_cc=0 --tau psi8_rotation=0.3315249"
for f in failure_diagnostics.py make_diag_npz.py name_cases.py $MLIST $FLIST outputs/busi_UKAN_seed2981/model.pth outputs/busi_UKAN_seed2981/percase_best.csv; do [ -f $f ] || { echo "缺少 $f"; exit 1; }; done
step() { echo; echo "===== $1　$(date '+%m/%d %H:%M')"; }

step "0. failure_diagnostics.py 自我測試"
python failure_diagnostics.py --selftest | tail -n 1 | grep -q PASSED || { echo "自我測試失敗"; exit 1; }
echo "PASSED"

step "1. seed2981（best）：機率圖、Ψ、梯度"
python make_diag_npz.py --name busi_UKAN_seed2981 --which best --features --grads --flag_list $FLIST || exit 1
python failure_diagnostics.py --npz $OUT/busi_UKAN_seed2981_best.npz --out-dir $OUT/seed2981_best --delta 0.5 $TAUS \
  --shape-features psi1_convex psi6_cc > $OUT/seed2981_best.txt || exit 1
python name_cases.py $OUT/busi_UKAN_seed2981_best.npz $OUT/seed2981_best/per_case.csv $MLIST || exit 1

for m in base2 shift; do
  step "2. $m（last）：機率圖"
  python make_diag_npz.py --name busi_UKAN_$m --which last || exit 1
  python failure_diagnostics.py --npz $OUT/busi_UKAN_${m}_last.npz --out-dir $OUT/${m}_last --delta 0.5 > $OUT/${m}_last.txt || exit 1
  python name_cases.py $OUT/busi_UKAN_${m}_last.npz $OUT/${m}_last/per_case.csv $MLIST || exit 1
done

step "3. 複製到 $DISC/busi"
if [ -d $DISC/.git ]; then
  B=$DISC/busi
  mkdir -p $B/code $B/results/diagnostics
  cp failure_diagnostics.py make_diag_npz.py name_cases.py run_diag.sh $B/code/
  for d in seed2981_best base2_last shift_last; do
    mkdir -p $B/results/diagnostics/$d
    cp $OUT/$d/per_case.csv $OUT/$d/summary.json $B/results/diagnostics/$d/
  done
  (cd $DISC && git add -A && git status --short)
else
  echo "找不到 $DISC，略過複製"
fi
step "全部完成"
for d in seed2981_best base2_last shift_last; do
  echo "--- $d"; python -c "import json;s=json.load(open('$OUT/$d/summary.json'));print('失敗',s['n_failures'],'/',s['n_cases'],' 類型',s['failure_types'],' recall@k（失敗）',s['recall_at_k_on_failures'])"
done