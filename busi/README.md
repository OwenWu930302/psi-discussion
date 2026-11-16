# BUSI：困難樣本與過擬合（討論用實驗）

**結論**
1. U-KAN 在 BUSI 上有過擬合：訓練時加入隨機平移，測試集 Dice 0.782 → 0.818（+0.036，p < 0.001），重訓一次結果相同。
2. 測試集中 Dice 接近 0 的 9 張困難影像，任何資料增強都只到 0.06–0.09：這是「沒見過的類型」，正則化救不回來。
3. 搬進訓練集後這 9 張能到 0.962，但只是死記：平移 12 px 就掉到 0.188。
4. Ψ8 加權在平移增強之上沒有額外效果（−0.008，p = 0.29）。

模型、切分、400 epoch、超參數、評估方式全部與 [U-KAN 官方](https://github.com/CUHK-AIM-Group/U-KAN) 相同，分析一律用最後一個 epoch（last）。
對照組 base2、base3、moved、Ψ 專家的逐張結果在 [psi-weighted-busi/results/percase](https://github.com/OwenWu930302/psi-weighted-busi/tree/main/results/percase)。

## 資料夾

| 路徑 | 內容 |
|---|---|
| [code/](code/) | 對 `train.py` 的修改（patch_*.py）、訓練腳本（run_*.sh）、擾動測試（perturb_busi.py）、分析（make_busi_results.py） |
| [results/percase/](results/percase/) | 本倉庫每個模型的逐張 Dice（best、last），格式同 psi-weighted-busi |
| [results/training_logs/](results/training_logs/) | 每個模型 400 epoch 的訓練紀錄 |
| [results/analysis/](results/analysis/) | 所有彙整表（下面每一節都註明用哪一個） |
| [results/examples/](results/examples/) | 那 9 張的預測圖：列 = shift、shift_moved、moved、base2；綠 = 標註、紅 = 預測、黃字 = Dice |

## 1. 我們改了什麼：訓練時即時的資料增強，沒有增加樣本

訓練集 517 張、測試集 130 張都沒變。每次讀取一張訓練影像時隨機變換（影像與遮罩同步），同一張影像每個 epoch 看到的版本都不同；測試時不做任何變換。
唯一動到樣本的是 `moved`、`shift_moved`：把那 9 張測試影像搬進訓練集（訓練 526、測試 121）。

| `train.py` 選項 | 作用 | 程式 |
|---|---|---|
| （官方） | 隨機旋轉 90° + 翻轉 | 原版 |
| `--shift_aug 0.125` | 隨機平移 ±32 px（p = 0.5，邊緣補 0） | [patch_shiftaug.py](code/patch_shiftaug.py) |
| `--scale_aug 0.1 --rotate_aug 15` | 再加縮放 ±10%、旋轉 ±15° | [patch_aug2.py](code/patch_aug2.py) |
| `--intensity_aug 0.2` | 再加亮度、對比 ±20%、gamma 80–120 | [patch_aug2.py](code/patch_aug2.py) |
| `--drop_rate 0.1` / `0.3` | KAN 層 Dropout（原版建立但沒有呼叫，接上） | patch_dropout.py（進行中） |

不給任何選項時，資料增強與網路輸出和原版逐位元相同（已驗證）。訓練指令 = 官方指令 + 選項：

```
python train.py --arch UKAN --dataset busi --input_w 256 --input_h 256 --name busi_UKAN_shift --data_dir ./inputs --shift_aug 0.125
```

## 2. 每個模型的分數 — [aug_BUSI_models.csv](results/analysis/aug_BUSI_models.csv)

| 模型 | 資料增強 | 測試集 130 | 一般 115 | 低分 15 | 那 9 張 | 訓練集 517 | 差距 |
|---|---|---|---|---|---|---|---|
| base2（對照組） | 官方 | 0.782 | 0.860 | 0.180 | 0.003 | 0.966 | 0.184 |
| base3（重訓） | 官方 | 0.791 | 0.874 | 0.156 | 0.006 | 0.967 | 0.175 |
| psi8w3n | 官方 + Ψ8 加權 | 0.804 | 0.882 | 0.203 | 0.016 | 0.959 | 0.155 |
| **shift** | + 平移 | **0.818** | 0.895 | 0.233 | 0.089 | 0.939 | 0.120 |
| **shift_r2（重訓）** | + 平移 | **0.818** | 0.895 | 0.225 | 0.070 | 0.936 | 0.118 |
| ssr | + 平移、縮放、旋轉 | 0.822 | 0.902 | 0.209 | 0.063 | 0.929 | 0.107 |
| ssr_int | + 平移、縮放、旋轉、強度 | 0.818 | 0.894 | 0.234 | 0.070 | 0.926 | 0.108 |
| psi8w3n_shift | + 平移 + Ψ8 加權 | 0.810 | 0.892 | 0.186 | 0.022 | 0.928 | 0.118 |

低分 15 = baseline 測試集 Dice < 0.5；那 9 張 = 其中 Dice < 0.2 者；差距 = 訓練集 − 測試集。
搬移模型（測試集只剩 121 張）：moved 那 9 張 0.962、shift_moved 0.789（`benign (116)` 仍 0.000）。

## 3. 成對比較 — [aug_BUSI_summary.csv](results/analysis/aug_BUSI_summary.csv)

測試集 130 張，逐張配對 Wilcoxon。

| 比較（control → model） | 測試集 Δ | p | 那 9 張 Δ |
|---|---|---|---|
| base2 → base3（不加增強的重訓雜訊） | +0.009 | 0.70 | +0.003 |
| shift → shift_r2（加平移的重訓雜訊） | −0.001 | 0.46 | −0.019 |
| base2 → shift | **+0.037** | **1.3e-4** | +0.086 |
| base3 → shift | **+0.027** | **0.0023** | +0.083 |
| base2 → shift_r2 | **+0.036** | **4.5e-5** | +0.067 |
| base3 → shift_r2 | **+0.027** | **0.0016** | +0.064 |
| shift → ssr（加縮放、旋轉） | +0.004 | 0.57 | −0.027 |
| ssr → ssr_int（加強度） | −0.004 | 0.68 | +0.008 |
| base2 → psi8w3n（Ψ8，無增強） | +0.022 | 0.027 | +0.013 |
| base3 → psi8w3n | +0.013 | 0.17 | +0.010 |
| shift → psi8w3n_shift（Ψ8，有平移） | −0.008 | 0.29 | −0.067 |
| shift_r2 → psi8w3n_shift | −0.008 | 0.095 | −0.048 |

逐張前後：[aug_BUSI_percase.csv](results/analysis/aug_BUSI_percase.csv)。

## 4. 為什麼是平移 — [sweep_BUSI_summary.csv](results/analysis/sweep_BUSI_summary.csv)

![平移掃描](results/analysis/shift_sweep_BUSI.png)

U-KAN 把影像縮小 32 倍（5 次 ×2），而官方增強只有翻轉和旋轉 90°，所以每張影像只在同一個 32 px 格子位置被看過。
不加平移增強時，平移量不是 32 的倍數 Dice 就下降，32、64 px 時幾乎回到原本；看過的 9 張下降得遠比一般訓練影像深。加了平移增強後曲線變平。

## 5. 擾動測試：看過的 9 張是學會還是死記 — [perturb_BUSI_summary.csv](results/analysis/perturb_BUSI_summary.csv)

影像和遮罩做同樣的變化後算 Dice；train_ref30 = 隨機 30 張一般訓練影像，用來量出變化本身的影響。

| 變化（輕度） | moved：9 張 | moved：一般 30 張 | base2：9 張（沒看過） |
|---|---|---|---|
| 不變化 | 0.962 | 0.969 | 0.003 |
| 翻轉、旋轉 90°（訓練時看過） | 0.910 | 0.949 | 0.025 |
| 平移 5% | 0.256 | 0.772 | 0.017 |
| 縮放 ±10% | 0.183 | 0.710 | 0.051 |
| 旋轉 ±10° | 0.030 | 0.744 | 0.010 |
| 亮度 ±15 | 0.951 | 0.964 | 0.024 |
| 對比 ±20% | 0.953 | 0.966 | 0.005 |
| 模糊 σ = 1 | 0.935 | 0.921 | 0.001 |

斑點雜訊（speckle）強度設太高，連一般影像都掉到 0.3，不採用。

## CSV 欄位（以實際一列說明）

- `aug_BUSI_models.csv`：`busi_UKAN_shift,517,flip+rot90 + shift32,0,none,130,0.8184,0.8949,0.2326,0.0889,no,0.9389,0.1204` = 模型、訓練張數、資料增強、Dropout、Ψ 加權、測試張數、測試集、一般、低分、那 9 張、9 張是否在訓練集、訓練集、差距。
- `aug_BUSI_summary.csv`：`shift,busi_UKAN_base2,busi_UKAN_shift,all,130,0.7818,0.8184,0.0366,78,48,1.3e-04` = 比較名稱、對照、模型、組別、張數、對照分數、模型分數、Δ、升的張數、降的張數、p。
- `aug_BUSI_percase.csv`：`noise_noaug,busi_UKAN_base2,busi_UKAN_base3,benign (116),test,yes,yes,0.0000,0.0000,0.0000` = 比較、對照、模型、影像、所在切分、是否低分、是否那 9 張、對照 Dice、模型 Dice、Δ。
- `perturb_BUSI_percase.csv`：`busi_UKAN_moved,nine,malignant (198),rotate,1,10,+,0.0303,7572,5241` = 模型、組別、影像、變化、強度（1 輕 / 2 中 / 3 重）、數值、方向、Dice、預測像素、標註像素。`sweep_BUSI_percase.csv` 另有 `prob_gt`（標註區域內的平均預測機率）。

## 重新產生分析

兩個倉庫 clone 在同一層，於 psi-discussion 根目錄執行：

```
python busi/code/make_busi_results.py --busi_repo ../psi-weighted-busi --disc .
```
