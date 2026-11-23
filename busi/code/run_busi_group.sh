#!/bin/bash
# BUSI 分組搬移 + 兩個不用訓練的補充（結果只放 psi-discussion）；可中斷後重跑
#   0a. 疊合圖：低分但不在那 9 張的 6 張，各模型的預測（看「每個模型都一樣低」是否為標註 / 影像問題）
#   0b. 梯度對齊改用 decoder5 + final（教授程式預設只用 final 的 17 個參數）
#   1–3. 分組搬移：那 9 張分 3 組（每組：1 張模型完全沒看到的、1 張 benign、1 張 malignant），
#        每次把另外 6 張搬進訓練集（設定與 moved 相同，只差搬幾張），看沒搬的 3 張會不會變好
#        → 會變好：看過同類型就能推廣（缺資料）；仍是 0：每張都得親自看過（死記）
#   4. 分析；5. 複製到 psi-discussion 並 git add（不 commit）
# 用法（Seg_UKAN 資料夾、ukan 環境）：nohup bash run_busi_group.sh > run_busi_group.log 2>&1 &
cd "$(dirname "$0")"
DISC=${DISC:-$HOME/桌面/論文/psi-discussion}
LISTS=${LISTS:-/mnt/data2t/BUSI/psi_lists}
MLIST=$LISTS/moved_busi_dice02.csv
FLIST=$LISTS/wcases_busi_psi8_rotation.csv
GRP_SPEC=${GRP_SPEC:-"benign (247)|benign (223)|malignant (198);benign (116)|benign (257)|malignant (105);benign (236)|benign (367)|malignant (106)"}
LOW6=${LOW6:-"benign (299),malignant (116),benign (131),benign (353),benign (408),benign (315)"}
TRAIN=${TRAIN:-python train.py}         # 測試用；正式執行不需設定
EXTRA=${EXTRA:-}                        # 測試用；正式執行不需設定
EPOCHS=${EPOCHS:-400}
GL=outputs/group_lists
for f in train.py predict_busi.py overlay_low.py analyze_group.py make_diag_npz.py failure_diagnostics.py name_cases.py $MLIST $FLIST; do
  [ -f "$f" ] || { echo "缺少 $f"; exit 1; }
done
step() { echo; echo "===== $1　$(date '+%m/%d %H:%M')"; }

step "0a. 疊合圖（6 張）"
python overlay_low.py --cases "$LOW6" || exit 1

step "0b. 梯度對齊（decoder5 + final）"
OUTD=outputs/diag
if [ -f $OUTD/seed2981_best_dec5/summary.json ]; then echo "已完成，略過"; else
  python make_diag_npz.py --name busi_UKAN_seed2981 --which best --features --grads --flag_list $FLIST --params decoder5,final \
    --out $OUTD/busi_UKAN_seed2981_best_dec5.npz || exit 1
  python failure_diagnostics.py --npz $OUTD/busi_UKAN_seed2981_best_dec5.npz --out-dir $OUTD/seed2981_best_dec5 --delta 0.5 \
    --tau psi1_convex=1.0512699 --tau psi6_cc=0 --tau psi8_rotation=0.3315249 --shape-features psi1_convex psi6_cc \
    > $OUTD/seed2981_best_dec5.txt || exit 1
  python name_cases.py $OUTD/busi_UKAN_seed2981_best_dec5.npz $OUTD/seed2981_best_dec5/per_case.csv $MLIST || exit 1
fi
python -c "import json;s=json.load(open('$OUTD/seed2981_best_dec5/summary.json'));print('失敗 vs 加權清單',s['alignment_on_failures']);print('失敗 vs 隨機對照',s['control_alignment_on_failures']);print('A>0 的失敗比例',s['frac_failures_with_A_positive'])"

step "分組清單"
mkdir -p $GL
python - "$MLIST" "$GRP_SPEC" "$GL" <<'PY' || exit 1
import sys, pandas as pd
nine = set(pd.read_csv(sys.argv[1], encoding='utf-8-sig').case.astype(str))
groups = [g.split('|') for g in sys.argv[2].split(';')]
allg = [c for g in groups for c in g]
assert sorted(allg) == sorted(nine), f'分組與那 9 張不符：{sorted(set(allg) ^ nine)}'
for k, g in enumerate(groups, 1):
    pd.DataFrame({'case': [c for c in sorted(nine) if c not in g]}).to_csv(f'{sys.argv[3]}/moved_g{k}.csv', index=False, encoding='utf-8-sig')
    print(f'g{k}：留在測試集 {g}，其餘 6 張搬進訓練集')
PY

i=0
for f in $GL/moved_g*.csv; do
  i=$((i + 1)); g=$(basename $f .csv); g=${g#moved_}; n=busi_UKAN_moved_$g
  step "$i. 訓練 $n（--move_to_train $f）"
  if [ -f outputs/$n/model_last.pth ] && [ "$(tail -n +2 outputs/$n/log.csv 2>/dev/null | wc -l)" -ge $EPOCHS ]; then
    echo "已完成，略過訓練"
  else
    $TRAIN --arch UKAN --dataset busi --input_w 256 --input_h 256 --name $n --data_dir ./inputs --move_to_train $f $EXTRA || exit 1
  fi
  for w in best last; do
    [ -f outputs/$n/percase_$w.csv ] && echo "已有 percase_$w.csv，略過預測" || python predict_busi.py --name $n --which $w || exit 1
  done
done

step "4. 分析"
python analyze_group.py --groups $GL --list $MLIST || exit 1

step "5. 複製到 $DISC/busi"
if [ -d $DISC/.git ]; then
  B=$DISC/busi
  mkdir -p $B/code $B/results/percase $B/results/training_logs $B/results/analysis $B/results/examples/low_other $B/results/diagnostics/seed2981_best_dec5 $B/results/group_lists
  cp overlay_low.py analyze_group.py make_diag_npz.py run_busi_group.sh $B/code/
  cp outputs/analysis_busi_overlay/*.png $B/results/examples/low_other/
  cp $OUTD/seed2981_best_dec5/per_case.csv $OUTD/seed2981_best_dec5/summary.json $B/results/diagnostics/seed2981_best_dec5/
  cp $GL/moved_g*.csv $B/results/group_lists/
  cp outputs/analysis_busi_group/group_BUSI_*.csv $B/results/analysis/
  for f in $GL/moved_g*.csv; do
    g=$(basename $f .csv); n=busi_UKAN_${g}
    [ -f outputs/$n/percase_last.csv ] || continue
    cp outputs/$n/percase_best.csv $B/results/percase/${n}_best.csv
    cp outputs/$n/percase_last.csv $B/results/percase/${n}_last.csv
    cp outputs/$n/log.csv $B/results/training_logs/${n}_log.csv
  done
  (cd $DISC && git add -A && git status --short)
else
  echo "找不到 $DISC，略過複製"
fi
step "全部完成"