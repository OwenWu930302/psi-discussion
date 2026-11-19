# BUSI：困難樣本與過擬合（討論用實驗）

**結論**
1. U-KAN 在 BUSI 上有過擬合：訓練時加入隨機平移，測試集 Dice 0.782 → 0.818（+0.036，p < 0.001），重訓一次結果相同。
2. 測試集中 Dice 接近 0 的 9 張困難影像，任何資料增強都只到 0.06–0.09：這是「沒見過的類型」，正則化救不回來。
3. 搬進訓練集後這 9 張能到 0.962，但只是死記：平移 12 px 就掉到 0.188。
4. 通用正則化都沒有顯著效果：Dropout 10% / 30%、斑點雜訊相對 base3 分別 +0.002、+0.005、+0.004（p = 0.80、0.39、0.83），也沒有改善對平移的敏感；加在平移增強之上也沒有幫助。
5. Ψ8 加權在平移增強之上沒有額外效果（−0.008，p = 0.29）。

模型、切分、400 epoch、超參數、評估方式全部與 [U-KAN 官方](https://github.com/CUHK-AIM-Group/U-KAN) 相同，分析一律用最後一個 epoch（last）。
對照組 base2、base3、moved、Ψ 專家的逐張結果在 [psi-weighted-busi/results/percase](https://github.com/OwenWu930302/psi-weighted-busi/tree/main/results/percase)。

## 0. 模型一覽（依類別）

★ = 報告用的主要模型。表中的 drop10 / drop30 在訓練資料夾與逐張檔名中是 drop1 / drop3（[aug_BUSI_models.csv](results/analysis/aug_BUSI_models.csv) 的 `train_name` 欄）。

| 類別 | 模型 | 做了什麼 / 回答的問題 | 訓練張數 |
|---|---|---|---|
| 對照組 | ★ `base2` | 原版 U-KAN | 517 |
| 對照組 | ★ `base3` | 原版重訓一次，量雜訊 | 517 |
| 搬移（動機） | ★ `moved` | 9 張搬進訓練集：看過就會嗎？ | 526 |
| 搬移（動機） | `shift_moved` | 9 張搬進訓練集 + 平移增強：還能死記嗎？ | 526 |
| 資料增強 | ★ `shift` | 隨機平移 ±32 px（主要結果） | 517 |
| 資料增強 | ★ `shift_r2` | shift 重訓一次，確認重現 | 517 |
| 資料增強 | `ssr` | 平移 + 縮放 ±10% + 旋轉 ±15° | 517 |
| 資料增強 | `ssr_int` | ssr + 亮度、對比、gamma | 517 |
| 通用正則化（Dropout、雜訊） | ★ `drop10` | Dropout 10% | 517 |
| 通用正則化（Dropout、雜訊） | ★ `drop30` | Dropout 30% | 517 |
| 通用正則化（Dropout、雜訊） | `drop30_r2` | drop30 重訓一次，確認重現（進行中） | 517 |
| 通用正則化（Dropout、雜訊） | ★ `speckle` | 斑點雜訊：每個像素 ×[0.9, 1.1] | 517 |
| 平移增強 + Dropout | `shift_drop10` | 平移增強 + Dropout 10% | 517 |
| 平移增強 + Dropout | `shift_drop30` | 平移增強 + Dropout 30% | 517 |
| Ψ 加權 | `psi8w3n` | Ψ8 加權 ×3（原研究方法） | 517 |
| Ψ 加權 | `psi8w3n_shift` | 平移增強 + Ψ8 加權 | 517 |

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
| `--noise_aug 0.1` | 斑點雜訊：每個像素 ×[0.9, 1.1]（p = 0.5，只改影像） | [patch_noise.py](code/patch_noise.py) |
| `--drop_rate 0.1` / `0.3` | KAN 層 Dropout（原版建立了但 forward 沒有呼叫，接上） | [patch_dropout.py](code/patch_dropout.py) |

不給任何選項時，資料增強與網路輸出和原版逐位元相同（已驗證）。訓練指令 = 官方指令 + 選項：

```
python train.py --arch UKAN --dataset busi --input_w 256 --input_h 256 --name busi_UKAN_shift --data_dir ./inputs --shift_aug 0.125
```

## 2. 每個模型的分數 — [aug_BUSI_models.csv](results/analysis/aug_BUSI_models.csv)

| 類別 | 模型 | 測試集 | 一般 115 | 低分 15 | 那 9 張 | 訓練集 517 | 差距 |
|---|---|---|---|---|---|---|---|
| 對照組 | `base2` | 0.782 | 0.860 | 0.180 | 0.003 | 0.966 | 0.184 |
| 對照組 | `base3` | 0.791 | 0.874 | 0.156 | 0.005 | 0.967 | 0.175 |
| 搬移（動機） | `moved` | 0.837\* | 0.853\* | 0.542\* | 0.962 | 0.968 | 0.130 |
| 搬移（動機） | `shift_moved` | 0.872\* | 0.895\* | 0.442\* | 0.789 | 0.936 | 0.064 |
| 資料增強 | `shift` | **0.818** | 0.895 | 0.233 | 0.089 | 0.939 | 0.120 |
| 資料增強 | `shift_r2` | **0.818** | 0.895 | 0.225 | 0.070 | 0.936 | 0.118 |
| 資料增強 | `ssr` | 0.822 | 0.902 | 0.209 | 0.062 | 0.929 | 0.107 |
| 資料增強 | `ssr_int` | 0.818 | 0.894 | 0.234 | 0.070 | 0.926 | 0.108 |
| 通用正則化（Dropout、雜訊） | `drop10` | 0.793 | 0.870 | 0.207 | 0.000 | 0.965 | 0.172 |
| 通用正則化（Dropout、雜訊） | `drop30` | 0.796 | 0.869 | 0.238 | 0.058 | 0.959 | 0.163 |
| 通用正則化（Dropout、雜訊） | `speckle` | 0.795 | 0.865 | 0.255 | 0.114 | 0.964 | 0.169 |
| 平移增強 + Dropout | `shift_drop10` | 0.815 | 0.895 | 0.204 | 0.076 | 0.935 | 0.120 |
| 平移增強 + Dropout | `shift_drop30` | 0.818 | 0.893 | 0.238 | 0.110 | 0.934 | 0.116 |
| Ψ 加權 | `psi8w3n` | 0.804 | 0.882 | 0.203 | 0.016 | 0.959 | 0.155 |
| Ψ 加權 | `psi8w3n_shift` | 0.810 | 0.892 | 0.186 | 0.022 | 0.928 | 0.118 |

測試集 = 130 張；低分 15 = baseline 測試集 Dice < 0.5；那 9 張 = 其中 Dice < 0.2 者；差距 = 訓練集 − 測試集。
\* 搬移模型的測試集只剩 121 張，那 9 張在訓練集（看過），不能直接和其他列比較。

## 3. 成對比較 — [aug_BUSI_summary.csv](results/analysis/aug_BUSI_summary.csv)

測試集逐張配對 Wilcoxon；Δ = 模型 − 對照。搬移比較用 nine（那 9 張）與 remaining（其餘 121 張）。

| 類別 | 比較（comparison） | 對照 → 模型 | 測試集 Δ | p | 那 9 張 Δ |
|---|---|---|---|---|---|
| 對照組 | `noise_noaug` | base2 → base3（不加增強的重訓雜訊） | +0.009 | 0.70 | +0.003 |
| 搬移（動機） | `moved_noaug` | base2 → moved | −0.003（121 張） | 0.69 | +0.960 |
| 搬移（動機） | `moved_shift` | shift → shift_moved | −0.001（121 張） | 0.95 | +0.700 |
| 搬移（動機） | `moved_shift` | moved → shift_moved | +0.035（121 張） | 3.1e-4 | −0.173 |
| 資料增強 | `noise_shift` | shift → shift_r2（加平移的重訓雜訊） | −0.001 | 0.46 | −0.019 |
| 資料增強 | `shift` | base2 → shift | **+0.037** | **1.3e-4** | +0.086 |
| 資料增強 | `shift` | base3 → shift | **+0.027** | **0.0023** | +0.083 |
| 資料增強 | `shift_r2` | base2 → shift_r2 | **+0.036** | **4.5e-5** | +0.067 |
| 資料增強 | `shift_r2` | base3 → shift_r2 | **+0.027** | **0.0016** | +0.064 |
| 資料增強 | `ssr` | shift → ssr | +0.004 | 0.57 | −0.026 |
| 資料增強 | `ssr` | shift_r2 → ssr | +0.004 | 0.80 | −0.007 |
| 資料增強 | `ssr_int` | ssr → ssr_int | −0.004 | 0.68 | +0.008 |
| 通用正則化（Dropout、雜訊） | `drop10` | base2 → drop10 | +0.011 | 0.32 | −0.003 |
| 通用正則化（Dropout、雜訊） | `drop10` | base3 → drop10 | +0.002 | 0.80 | −0.005 |
| 通用正則化（Dropout、雜訊） | `drop30` | base2 → drop30 | +0.015 | 0.14 | +0.055 |
| 通用正則化（Dropout、雜訊） | `drop30` | base3 → drop30 | +0.005 | 0.39 | +0.052 |
| 通用正則化（Dropout、雜訊） | `speckle` | base2 → speckle | +0.013 | 0.54 | +0.111 |
| 通用正則化（Dropout、雜訊） | `speckle` | base3 → speckle | +0.004 | 0.83 | +0.108 |
| 平移增強 + Dropout | `shift_drop10` | shift → shift_drop10 | −0.004 | 0.11 | −0.013 |
| 平移增強 + Dropout | `shift_drop10` | shift_r2 → shift_drop10 | −0.003 | 0.12 | +0.007 |
| 平移增強 + Dropout | `shift_drop30` | shift → shift_drop30 | −0.001 | 0.75 | +0.021 |
| 平移增強 + Dropout | `shift_drop30` | shift_r2 → shift_drop30 | 0.000 | 0.17 | +0.040 |
| Ψ 加權 | `psi8_noaug` | base2 → psi8w3n | **+0.022** | **0.03** | +0.013 |
| Ψ 加權 | `psi8_noaug` | base3 → psi8w3n | +0.013 | 0.17 | +0.010 |
| Ψ 加權 | `psi8_shift` | shift → psi8w3n_shift | −0.008 | 0.29 | −0.067 |
| Ψ 加權 | `psi8_shift` | shift_r2 → psi8w3n_shift | −0.008 | 0.09 | −0.048 |

完整（含一般、低分、訓練集各組）與逐張前後：[aug_BUSI_summary.csv](results/analysis/aug_BUSI_summary.csv)、[aug_BUSI_percase.csv](results/analysis/aug_BUSI_percase.csv)。

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

擾動測試的斑點雜訊（speckle）強度設太高，連一般影像都掉到 0.3，不採用（和第 6 節訓練時加的雜訊無關）。

## 6. 通用正則化（Dropout、雜訊）與平移增強的比較

Dropout 與斑點雜訊都只讓訓練集分數略降（0.966 → 0.959、0.964），差距從 0.184 縮到 0.163、0.169；平移增強縮到 0.120。
相對 base3，測試集 drop10 +0.002（p = 0.80）、drop30 +0.005（p = 0.39）、speckle +0.004（p = 0.83），都在重訓雜訊範圍內。
低分 15 張有一點跡象（speckle 比 base2 +0.075，p = 0.02；比 base3 +0.100，p = 0.13），但只訓練一次，需要重訓確認。
Dropout 也沒有改善對平移的敏感：一般訓練影像平移 12 px 時，base2 0.794、drop10 0.814、drop30 0.821，shift 0.939（[sweep_BUSI_summary.csv](results/analysis/sweep_BUSI_summary.csv)，`group = train_ref30`）。
針對已知弱點的正則化（平移增強）有效；通用正則化（Dropout、雜訊）與強度類增強都沒有顯著效果。

## 7. 低分 15 張逐張 — [low_dice_BUSI_wide.csv](results/analysis/low_dice_BUSI_wide.csv)

每個模型對 15 張低分影像（baseline 測試集 Dice < 0.5）的 Dice，全部用 last；baseline = seed2981 best（定義低分用）。moved、shift_moved 的「那 9 張」在訓練集（看過）。

| 影像 | 那 9 張 | baseline | base2 | base3 | moved | shift_moved | shift | shift_r2 | ssr | ssr_int | drop10 | drop30 | speckle | shift_drop10 | shift_drop30 | psi8w3n | psi8w3n_shift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `benign (247)` | 是 | 0.00 | 0.00 | 0.00 | 0.97 | 0.68 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `benign (367)` | 是 | 0.00 | 0.00 | 0.00 | 0.98 | 0.96 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `benign (116)` | 是 | 0.00 | 0.00 | 0.00 | 0.96 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `benign (236)` | 是 | 0.00 | 0.00 | 0.00 | 0.99 | 0.91 | 0.00 | 0.16 | 0.20 | 0.20 | 0.00 | 0.00 | 0.00 | 0.30 | 0.00 | 0.00 | 0.00 |
| `malignant (198)` | 是 | 0.01 | 0.01 | 0.05 | 0.97 | 0.90 | 0.02 | 0.05 | 0.28 | 0.29 | 0.00 | 0.00 | 0.00 | 0.06 | 0.13 | 0.05 | 0.16 |
| `benign (257)` | 是 | 0.02 | 0.00 | 0.00 | 0.96 | 0.95 | 0.32 | 0.16 | 0.00 | 0.00 | 0.00 | 0.45 | 0.42 | 0.25 | 0.41 | 0.00 | 0.00 |
| `benign (223)` | 是 | 0.03 | 0.01 | 0.00 | 0.97 | 0.94 | 0.34 | 0.02 | 0.00 | 0.00 | 0.00 | 0.07 | 0.51 | 0.06 | 0.43 | 0.00 | 0.00 |
| `malignant (105)` | 是 | 0.05 | 0.00 | 0.00 | 0.92 | 0.89 | 0.07 | 0.20 | 0.00 | 0.04 | 0.00 | 0.00 | 0.03 | 0.01 | 0.00 | 0.01 | 0.00 |
| `malignant (106)` | 是 | 0.08 | 0.00 | 0.00 | 0.94 | 0.87 | 0.05 | 0.04 | 0.08 | 0.11 | 0.00 | 0.00 | 0.06 | 0.02 | 0.01 | 0.09 | 0.03 |
| `benign (353)` |  | 0.13 | 0.65 | 0.28 | 0.89 | 0.87 | 0.91 | 0.91 | 0.91 | 0.89 | 0.85 | 0.88 | 0.68 | 0.92 | 0.90 | 0.84 | 0.90 |
| `benign (299)` |  | 0.17 | 0.26 | 0.27 | 0.25 | 0.26 | 0.25 | 0.26 | 0.26 | 0.24 | 0.25 | 0.26 | 0.25 | 0.26 | 0.28 | 0.26 | 0.24 |
| `malignant (116)` |  | 0.24 | 0.23 | 0.23 | 0.22 | 0.23 | 0.25 | 0.26 | 0.25 | 0.22 | 0.23 | 0.24 | 0.23 | 0.22 | 0.25 | 0.23 | 0.24 |
| `benign (408)` |  | 0.25 | 0.27 | 0.32 | 0.78 | 0.30 | 0.20 | 0.20 | 0.21 | 0.28 | 0.55 | 0.43 | 0.31 | 0.04 | 0.21 | 0.21 | 0.22 |
| `benign (315)` |  | 0.41 | 0.82 | 0.71 | 0.66 | 0.56 | 0.63 | 0.67 | 0.53 | 0.82 | 0.76 | 0.74 | 0.86 | 0.51 | 0.50 | 0.89 | 0.55 |
| `benign (131)` |  | 0.48 | 0.45 | 0.47 | 0.45 | 0.43 | 0.45 | 0.44 | 0.42 | 0.42 | 0.47 | 0.50 | 0.47 | 0.43 | 0.43 | 0.48 | 0.44 |

15 張可分成三類：

1. **位置找錯（那 9 張）**：沒看過的模型幾乎都是 0；任何資料增強、Dropout 或雜訊最高只到 0.51，而且不同模型、不同次訓練改善的不是同一張。只有搬進訓練集（moved）才到 0.92–0.99。
2. **每個模型都一樣低（3 張）**：`benign (299)` 約 0.25、`malignant (116)` 約 0.23、`benign (131)` 約 0.44，所有模型幾乎相同，連 moved 也一樣。換模型、換訓練方式都不變，問題可能在影像或標註本身（需要看圖確認）。
3. **不穩定（3 張）**：`benign (353)`、`benign (408)`、`benign (315)` 在不同模型之間差很多（例如 353：base2 0.65、base3 0.28、所有平移增強模型約 0.90；408：0.04–0.78）。

每個模型相對 base2 的平均前後（low_dice 15 張、nine 9 張、low_other 6 張）：[low_dice_BUSI_summary.csv](results/analysis/low_dice_BUSI_summary.csv)；逐張前後：[low_dice_BUSI_percase.csv](results/analysis/low_dice_BUSI_percase.csv)。

### 7.1 低分但不在那 9 張的 6 張

沒有任何模型能整體提升這 6 張：14 個模型相對 base2 的平均變化在 −0.065 到 +0.096 之間，p 全部 ≥ 0.16；光是 base2 重訓成 base3 就差 −0.065。

每張影像在 14 個模型的情況（[low_other_BUSI_case_summary.csv](results/analysis/low_other_BUSI_case_summary.csv)，前 = base2）：

| 影像 | base2 | base3 | 14 個模型平均 | 最低–最高 | 升 / 降（模型數） | 解讀 |
|---|---|---|---|---|---|---|
| `benign (353)` | 0.65 | 0.28 | 0.83 | 0.28–0.92 | 13 / 1 | 被改善：所有平移增強模型都約 0.87–0.92 |
| `benign (408)` | 0.27 | 0.32 | 0.30 | 0.04–0.78 | 7 / 7 | 雜訊：每次訓練差很多，沒有固定方向 |
| `malignant (116)` | 0.23 | 0.23 | 0.24 | 0.22–0.26 | 6 / 8 | 不動：所有模型都約 0.23 |
| `benign (299)` | 0.26 | 0.27 | 0.26 | 0.24–0.28 | 7 / 7 | 不動：所有模型都約 0.26 |
| `benign (131)` | 0.45 | 0.47 | 0.45 | 0.42–0.50 | 5 / 9 | 不動：所有模型都約 0.45 |
| `benign (315)` | 0.82 | 0.71 | 0.67 | 0.50–0.89 | 2 / 12 | 變差：平移增強模型降到 0.50–0.67 |

- 每個模型的平均前後：[low_other_BUSI_summary.csv](results/analysis/low_other_BUSI_summary.csv)
- 逐張前後：[low_other_BUSI_percase.csv](results/analysis/low_other_BUSI_percase.csv)
- 一張一列、每個模型一欄：[low_other_BUSI_wide.csv](results/analysis/low_other_BUSI_wide.csv)

## CSV 欄位（以實際一列說明）

- `aug_BUSI_models.csv`：`augmentation,busi_UKAN_shift,yes,busi_UKAN_shift,517,flip+rot90 + shift32,0.0000,none,130,0.8184,0.8949,0.2326,0.0889,no,0.9389,0.1204` = 類別、模型（表中名稱）、是否主要模型、訓練資料夾名稱、訓練張數、資料增強、Dropout、Ψ 加權、測試張數、測試集、一般、低分、那 9 張、9 張是否在訓練集、訓練集、差距。
- `aug_BUSI_summary.csv`：`augmentation,shift,busi_UKAN_base2,busi_UKAN_shift,all,130,0.7818,0.8184,0.0366,78,48,1.3e-04` = 模型類別、比較名稱、對照、模型、組別、張數、對照分數、模型分數、Δ、升的張數、降的張數、p。
- `aug_BUSI_percase.csv`：`noise_noaug,busi_UKAN_base2,busi_UKAN_base3,benign (116),test,yes,yes,0.0000,0.0000,0.0000` = 比較、對照、模型、影像、所在切分、是否低分、是否那 9 張、對照 Dice、模型 Dice、Δ。
- `low_dice_BUSI_percase.csv`：`control,busi_UKAN_base3,benign (223),test,yes,0.0289,0.0100,0.0000,-0.0100` = 模型類別、模型、影像、該模型中這張在測試集還是訓練集、是否那 9 張、baseline Dice、對照組 base2 的 Dice、該模型的 Dice、差。
- `low_dice_BUSI_summary.csv`：每個模型一組三列（low_dice、nine、low_other），第一欄為模型類別，欄位同 aug_BUSI_summary，另有 `n_above_05` = 該模型 Dice ≥ 0.5 的張數。
- `low_other_BUSI_case_summary.csv`：`benign (353),0.1256,0.6494,0.2837,14,0.8299,0.2837,0.9176,0.6340,0.1805,13,1,13` = 影像、baseline（seed2981 的 best epoch，用來定義低分）、base2 與 base3（兩次原版訓練的 last）、模型數（base2 以外的 14 個）、這些模型的平均 / 最低 / 最高 / 最高−最低、相對 base2 的平均變化、比 base2 高的模型數、比 base2 低的模型數、Dice ≥ 0.5 的模型數。`low_other_BUSI_percase.csv`、`_wide.csv`、`_summary.csv` 的欄位同 low_dice 的對應檔。
- `perturb_BUSI_percase.csv`：`busi_UKAN_moved,nine,malignant (198),rotate,1,10,+,0.0303,7572,5241` = 模型、組別、影像、變化、強度（1 輕 / 2 中 / 3 重）、數值、方向、Dice、預測像素、標註像素。`sweep_BUSI_percase.csv` 另有 `prob_gt`（標註區域內的平均預測機率）。

## 重新產生分析

兩個倉庫 clone 在同一層，於 psi-discussion 根目錄執行：

```
python busi/code/make_busi_results.py --busi_repo ../psi-weighted-busi --disc .
```
