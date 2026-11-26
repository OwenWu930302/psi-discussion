#!/bin/bash
# BUSI 追加實驗（結果只放 psi-discussion）；可中斷後重跑
#   1 busi_UKAN_drop3_r2 ：與 busi_UKAN_drop3 完全相同（Dropout 0.3），再訓練一次 → 確認 Dropout 的效果是否重現
#   2 busi_UKAN_speckle  ：訓練時加斑點雜訊（每個像素 ×[0.9, 1.1]，p = 0.5）→ 教授建議的「加雜訊」
#   其餘設定與原作者相同（切分、400 epoch、batch 8、Adam）；9 張都留在測試集
#   跑完：逐張結果、訓練紀錄、程式複製到 psi-discussion/busi/，重新產生 results/analysis/，git add（不 commit）
# 用法（Seg_UKAN 資料夾、ukan 環境）：nohup bash run_busi_more2.sh > run_busi_more2.log 2>&1 &
cd "$(dirname "$0")"
MODELS=${MODELS:-drop3_r2 speckle}
DISC=${DISC:-$HOME/桌面/論文/psi-discussion}
BUSIREPO=${BUSIREPO:-$HOME/桌面/論文/psi-weighted-ukan}
TRAIN=${TRAIN:-python train.py}         # 測試用；正式執行不需設定
EXTRA=${EXTRA:-}                        # 測試用；正式執行不需設定
EPOCHS=${EPOCHS:-400}                   # 判斷訓練是否完成（官方預設 400）
declare -A ARGS=(
  [drop3_r2]="--drop_rate 0.3"
  [speckle]="--noise_aug 0.1"
)
for f in train.py archs.py patch_noise.py predict_busi.py make_busi_results.py; do
  [ -f $f ] || { echo "缺少 $f"; exit 1; }
done
grep -q "\[Dropout\]" archs.py && grep -q "\[Dropout\]" train.py || { echo "尚未執行 patch_dropout.py"; exit 1; }
grep -q "\[強度增強\]" train.py || { echo "尚未執行 patch_aug2.py"; exit 1; }
[ -d $DISC/.git ] || { echo "找不到 $DISC（git 倉庫）"; exit 1; }
[ -d $BUSIREPO/results/percase ] || { echo "找不到 $BUSIREPO/results/percase"; exit 1; }
for k in $MODELS; do [ -n "${ARGS[$k]}" ] || { echo "不認得的模型 $k"; exit 1; }; done
step() { echo; echo "===== $1　$(date '+%m/%d %H:%M')"; }

step "1. train.py 加入 --noise_aug"
python patch_noise.py || exit 1
grep -q "\[雜訊增強\]" train.py || { echo "train.py 修改失敗"; exit 1; }

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

step "複製到 $DISC/busi 並重新產生分析"
B=$DISC/busi
mkdir -p $B/code $B/results/percase $B/results/training_logs
cp patch_noise.py run_busi_more2.sh make_busi_results.py $B/code/
for k in $MODELS; do
  n=busi_UKAN_$k
  cp outputs/$n/percase_best.csv $B/results/percase/${n}_best.csv
  cp outputs/$n/percase_last.csv $B/results/percase/${n}_last.csv
  cp outputs/$n/log.csv $B/results/training_logs/${n}_log.csv
done
(cd $DISC && python busi/code/make_busi_results.py --busi_repo $BUSIREPO --disc .) || exit 1
(cd $DISC && git add -A && git status --short)
step "全部完成"