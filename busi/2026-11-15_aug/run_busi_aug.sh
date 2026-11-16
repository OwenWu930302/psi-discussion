#!/bin/bash
# BUSI 資料增強追加實驗（結果只放 psi-discussion，不放 psi-weighted-busi）；依優先順序訓練，可中斷後重跑
#   1 busi_UKAN_shift_r2       ：與 busi_UKAN_shift 完全相同，再訓練一次（確認效果、平移增強下的雜訊）
#   2 busi_UKAN_ssr            ：平移 ±32 px + 縮放 ±10% + 旋轉 ±15°（機率 0.5）
#   3 busi_UKAN_ssr_int        ：2 + 亮度 / 對比 ±20%、gamma 80–120（各機率 0.5）
#   4 busi_UKAN_psi8w3n_shift  ：Ψ8 專家（×3、÷n）+ 平移增強；對照組 = shift、shift_r2
#   其餘設定與原作者相同（切分、400 epoch、batch 8、Adam）；9 張都留在測試集
# 只訓練其中幾個：MODELS="shift_r2 ssr" nohup bash run_busi_aug.sh > run_busi_aug.log 2>&1 &
# 用法（Seg_UKAN 資料夾、ukan 環境）：nohup bash run_busi_aug.sh > run_busi_aug.log 2>&1 &
cd "$(dirname "$0")"
LISTS=${LISTS:-/mnt/data2t/BUSI/psi_lists}
MLIST=$LISTS/moved_busi_dice02.csv
WLIST=$LISTS/wcases_busi_psi8_rotation.csv
MODELS=${MODELS:-shift_r2 ssr ssr_int psi8w3n_shift}
DISC=${DISC:-$HOME/桌面/論文/psi-discussion}
DEST=$DISC/busi/${TAG:-2026-11-15_aug}
TRAIN=${TRAIN:-python train.py}         # 測試用；正式執行不需設定
EXTRA=${EXTRA:-}                        # 測試用；正式執行不需設定
EPOCHS=${EPOCHS:-400}                   # 判斷訓練是否完成（官方預設 400）
REF=${REF:-30}                          # 擾動測試的一般訓練影像張數
SH="--shift_aug 0.125"
declare -A ARGS=(
  [shift_r2]="$SH"
  [ssr]="$SH --scale_aug 0.1 --rotate_aug 15"
  [ssr_int]="$SH --scale_aug 0.1 --rotate_aug 15 --intensity_aug 0.2"
  [psi8w3n_shift]="$SH --wcases $WLIST --wfactor 3.0 --wnorm n"
)
for f in train.py patch_aug2.py predict_busi.py perturb_busi.py analyze_aug.py $MLIST $WLIST; do
  [ -f $f ] || { echo "缺少 $f"; exit 1; }
done
grep -q "\[平移增強\]" train.py || { echo "train.py 尚未執行 patch_shiftaug.py"; exit 1; }
grep -q "set', default='main'" perturb_busi.py || { echo "perturb_busi.py 不是含 --set sweep 的新版"; exit 1; }
for k in $MODELS; do [ -n "${ARGS[$k]}" ] || { echo "不認得的模型 $k"; exit 1; }; done
step() { echo; echo "===== $1　$(date '+%m/%d %H:%M')"; }

step "1. train.py 加入 --scale_aug、--rotate_aug、--intensity_aug"
python patch_aug2.py || exit 1
grep -q "\[強度增強\]" train.py || { echo "train.py 修改失敗"; exit 1; }

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
PM=$(for k in shift shift_r2 ssr ssr_int psi8w3n_shift base2; do [ -f outputs/busi_UKAN_$k/model_last.pth ] && printf "busi_UKAN_%s," $k; done)
PM=${PM%,}; echo "模型：$PM"
python perturb_busi.py --models $PM --list $MLIST --ref $REF --out outputs/analysis_busi_aug/perturb || exit 1
python perturb_busi.py --models $PM --list $MLIST --ref $REF --set sweep --out outputs/analysis_busi_aug/perturb_sweep || exit 1

step "分析"
python analyze_aug.py --list $MLIST --wlist $WLIST || exit 1

step "複製到 $DEST"
if [ -d $DISC/.git ]; then
  mkdir -p $DEST/percase $DEST/training_logs
  cp patch_aug2.py run_busi_aug.sh analyze_aug.py $DEST/
  cp outputs/analysis_busi_aug/aug_*.csv $DEST/
  cp -r outputs/analysis_busi_aug/perturb outputs/analysis_busi_aug/perturb_sweep $DEST/
  for k in $MODELS; do
    n=busi_UKAN_$k
    [ -f outputs/$n/percase_last.csv ] || continue
    cp outputs/$n/percase_best.csv $DEST/percase/${n}_best.csv
    cp outputs/$n/percase_last.csv $DEST/percase/${n}_last.csv
    cp outputs/$n/log.csv $DEST/training_logs/${n}_log.csv
  done
  [ -f run_busi_aug.log ] && cp run_busi_aug.log $DEST/
  (cd $DISC && git add -A && git status --short)
else
  echo "找不到 $DISC（git 倉庫），略過複製"
fi
step "全部完成"