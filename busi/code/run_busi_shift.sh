#!/bin/bash
# BUSI 平移增強實驗（結果只放 psi-discussion，不放 psi-weighted-ukan）
#   A busi_UKAN_shift       ：隨機平移 ±32 px（機率 0.5），9 張留在測試集
#   B busi_UKAN_shift_moved ：同上，9 張移到訓練集
#   其餘設定與 base2 完全相同（原作者指令 + --shift_aug）
#   1. train.py 加入 --shift_aug（patch_shiftaug.py）  2. 訓練 A → 預測  3. 訓練 B → 預測
#   4. 擾動測試（main、sweep）  5. 分析  6. 複製到 psi-discussion 並 git add（不 commit）
# 可中斷後重跑：已完成的訓練與預測會略過
# 用法（Seg_UKAN 資料夾、ukan 環境）：nohup bash run_busi_shift.sh > run_busi_shift.log 2>&1 &
cd "$(dirname "$0")"
LISTS=${LISTS:-/mnt/data2t/BUSI/psi_lists}
MLIST=$LISTS/moved_busi_dice02.csv
SHIFT=${SHIFT:-0.125}
DISC=${DISC:-$HOME/桌面/論文/psi-discussion}
DEST=$DISC/busi/${TAG:-2026-11-14_shiftaug}
TRAIN=${TRAIN:-python train.py}         # 測試用；正式執行不需設定
EXTRA=${EXTRA:-}                        # 測試用；正式執行不需設定
EPOCHS=${EPOCHS:-400}                   # 判斷訓練是否完成（官方預設 400）
REF=${REF:-30}                          # 一般訓練影像張數（擾動測試的參考組）
A=busi_UKAN_shift
B=busi_UKAN_shift_moved
for f in train.py patch_shiftaug.py predict_busi.py perturb_busi.py analyze_shift.py $MLIST; do
  [ -f $f ] || { echo "缺少 $f"; exit 1; }
done
grep -q "set', default='main'" perturb_busi.py || { echo "perturb_busi.py 不是含 --set sweep 的新版"; exit 1; }
step() { echo; echo "===== $1　$(date '+%m/%d %H:%M')"; }

train() {   # $1 = 名稱，其餘 = 額外參數
  local n=$1; shift
  if [ -f outputs/$n/model_last.pth ] && [ "$(tail -n +2 outputs/$n/log.csv 2>/dev/null | wc -l)" -ge $EPOCHS ]; then
    echo "$n 已完成，略過訓練"
  else
    $TRAIN --arch UKAN --dataset busi --input_w 256 --input_h 256 --name $n --data_dir ./inputs --shift_aug $SHIFT "$@" $EXTRA || exit 1
  fi
  for w in best last; do
    [ -f outputs/$n/percase_$w.csv ] && echo "已有 $n/percase_$w.csv，略過預測" || python predict_busi.py --name $n --which $w || exit 1
  done
}

step "1. train.py 加入 --shift_aug"
python patch_shiftaug.py || exit 1
grep -q "\[平移增強\]" train.py || { echo "train.py 修改失敗"; exit 1; }

step "2. 訓練 A $A（9 張在測試集）"
train $A

step "3. 訓練 B $B（9 張在訓練集）"
train $B --move_to_train $MLIST

step "4. 擾動測試"
MODELS=$(for m in $A $B busi_UKAN_moved busi_UKAN_base2; do [ -f outputs/$m/model_last.pth ] && printf "%s," $m; done)
MODELS=${MODELS%,}
echo "模型：$MODELS"
python perturb_busi.py --models $MODELS --list $MLIST --ref $REF --out outputs/analysis_busi_shift/perturb || exit 1
python perturb_busi.py --models $MODELS --list $MLIST --ref $REF --set sweep --out outputs/analysis_busi_shift/perturb_sweep || exit 1

step "5. 分析"
python analyze_shift.py --list $MLIST || exit 1

step "6. 複製到 $DEST"
if [ -d $DISC/.git ]; then
  mkdir -p $DEST/percase $DEST/training_logs
  cp patch_shiftaug.py run_busi_shift.sh analyze_shift.py perturb_busi.py $DEST/
  cp outputs/analysis_busi_shift/shift_*.csv $DEST/
  cp -r outputs/analysis_busi_shift/perturb outputs/analysis_busi_shift/perturb_sweep $DEST/
  for m in $A $B; do
    cp outputs/$m/percase_best.csv $DEST/percase/${m}_best.csv
    cp outputs/$m/percase_last.csv $DEST/percase/${m}_last.csv
    cp outputs/$m/log.csv $DEST/training_logs/${m}_log.csv
  done
  [ -f run_busi_shift.log ] && cp run_busi_shift.log $DEST/
  (cd $DISC && git add -A && git status --short)
else
  echo "找不到 $DISC（git 倉庫），略過複製"
fi
step "全部完成"