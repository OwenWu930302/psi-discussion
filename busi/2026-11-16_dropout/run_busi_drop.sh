#!/bin/bash
# BUSI Dropout 實驗（結果只放 psi-discussion，不放 psi-weighted-busi）；依優先順序訓練，可中斷後重跑
#   U-KAN 原版的 KANLayer 建立了 Dropout 但沒有使用（drop_rate 無作用）→ patch_dropout.py 接上（只在 4 個 KAN 區塊）
#   1 busi_UKAN_drop1        ：Dropout 0.1（無資料增強）        對照組 base2、base3
#   2 busi_UKAN_shift_drop1  ：Dropout 0.1 + 平移增強 ±32 px     對照組 shift、shift_r2
#   3 busi_UKAN_drop3        ：Dropout 0.3（無資料增強）
#   4 busi_UKAN_shift_drop3  ：Dropout 0.3 + 平移增強
#   其餘設定與原作者相同（切分、400 epoch、batch 8、Adam）；9 張都留在測試集
# 只訓練其中幾個：MODELS="drop1 shift_drop1" nohup bash run_busi_drop.sh > run_busi_drop.log 2>&1 &
# 用法（Seg_UKAN 資料夾、ukan 環境）：nohup bash run_busi_drop.sh > run_busi_drop.log 2>&1 &
cd "$(dirname "$0")"
LISTS=${LISTS:-/mnt/data2t/BUSI/psi_lists}
MLIST=$LISTS/moved_busi_dice02.csv
MODELS=${MODELS:-drop1 shift_drop1 drop3 shift_drop3}
DISC=${DISC:-$HOME/桌面/論文/psi-discussion}
DEST=$DISC/busi/${TAG:-2026-11-16_dropout}
TRAIN=${TRAIN:-python train.py}         # 測試用；正式執行不需設定
EXTRA=${EXTRA:-}                        # 測試用；正式執行不需設定
EPOCHS=${EPOCHS:-400}                   # 判斷訓練是否完成（官方預設 400）
REF=${REF:-30}                          # 擾動測試的一般訓練影像張數
SH="--shift_aug 0.125"
declare -A ARGS=(
  [drop1]="--drop_rate 0.1"
  [shift_drop1]="$SH --drop_rate 0.1"
  [drop3]="--drop_rate 0.3"
  [shift_drop3]="$SH --drop_rate 0.3"
)
for f in train.py archs.py patch_dropout.py predict_busi.py perturb_busi.py analyze_drop.py $MLIST; do
  [ -f $f ] || { echo "缺少 $f"; exit 1; }
done
grep -q "\[平移增強\]" train.py || { echo "train.py 尚未執行 patch_shiftaug.py"; exit 1; }
grep -q "set', default='main'" perturb_busi.py || { echo "perturb_busi.py 不是含 --set sweep 的新版"; exit 1; }
for k in $MODELS; do [ -n "${ARGS[$k]}" ] || { echo "不認得的模型 $k"; exit 1; }; done
step() { echo; echo "===== $1　$(date '+%m/%d %H:%M')"; }

step "1. archs.py、train.py 加入 Dropout"
python patch_dropout.py || exit 1
grep -q "\[Dropout\]" train.py && grep -q "\[Dropout\]" archs.py || { echo "修改失敗"; exit 1; }

i=1
for k in $MODELS; do
  i=$((i + 1)); n=busi_UKAN_$k
  step "$i. 訓練 $n（${ARGS[$k]}）"
  if [ -f outputs/$n/model_last.pth ] && [ "$(tail -n +2 outputs/$n/log.csv 2>/dev/null | wc -l)" -ge $EPOCHS ]; then
    echo "已完成，略過訓練"
  else
    $TRAIN --arch UKAN --dataset busi --input_w 256 --input_h 256 --name $n --data_dir ./inputs ${ARGS[$k]} $EXTRA || exit 1
  fi
  for w in best last; do
    [ -f outputs/$n/percase_$w.csv ] && echo "已有 percase_$w.csv，略過預測" || python predict_busi.py --name $n --which $w || exit 1
  done
done

step "擾動測試"
PM=$(for k in drop1 shift_drop1 drop3 shift_drop3 shift base2; do [ -f outputs/busi_UKAN_$k/model_last.pth ] && printf "busi_UKAN_%s," $k; done)
PM=${PM%,}; echo "模型：$PM"
python perturb_busi.py --models $PM --list $MLIST --ref $REF --out outputs/analysis_busi_drop/perturb || exit 1
python perturb_busi.py --models $PM --list $MLIST --ref $REF --set sweep --out outputs/analysis_busi_drop/perturb_sweep || exit 1

step "分析"
python analyze_drop.py --list $MLIST || exit 1

step "複製到 $DEST"
if [ -d $DISC/.git ]; then
  mkdir -p $DEST/percase $DEST/training_logs
  cp patch_dropout.py run_busi_drop.sh analyze_drop.py $DEST/
  cp outputs/analysis_busi_drop/drop_*.csv $DEST/
  cp -r outputs/analysis_busi_drop/perturb outputs/analysis_busi_drop/perturb_sweep $DEST/
  for k in $MODELS; do
    n=busi_UKAN_$k
    [ -f outputs/$n/percase_last.csv ] || continue
    cp outputs/$n/percase_best.csv $DEST/percase/${n}_best.csv
    cp outputs/$n/percase_last.csv $DEST/percase/${n}_last.csv
    cp outputs/$n/log.csv $DEST/training_logs/${n}_log.csv
  done
  [ -f run_busi_drop.log ] && cp run_busi_drop.log $DEST/
  (cd $DISC && git add -A && git status --short)
else
  echo "找不到 $DISC（git 倉庫），略過複製"
fi
step "全部完成"