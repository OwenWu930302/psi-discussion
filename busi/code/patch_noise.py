"""
在 U-KAN train.py 加入 --noise_aug（訓練時的斑點雜訊增強；[雜訊增強]）
  需先執行過 patch_aug2.py（含 [強度增強]）
  --noise_aug a：MultiplicativeNoise，每個像素乘上 [1−a, 1+a] 的均勻亂數（三個通道相同 = 灰階斑點雜訊），p = 0.5；只改影像，不改遮罩
                 a = 0.1 → 雜訊標準差約 0.058（相當於擾動測試 speckle 的 0.06）
  預設 0：不給時 train_transform 與修改前完全相同
可重複執行（已修改過會略過）；找不到預期的程式碼時不修改並報錯
用法（Seg_UKAN 資料夾）：python patch_noise.py
"""
import sys

F = 'train.py'
s = open(F, encoding='utf-8').read()
if '[雜訊增強]' in s:
    print('train.py 已含 [雜訊增強]，略過')
    sys.exit()
if '[強度增強]' not in s:
    sys.exit('train.py 尚未執行 patch_aug2.py，未修改')
A1 = "    parser.add_argument('--intensity_aug', default=0.0, type=float, help='[強度增強] 亮度/對比 ±比例 + gamma；0 = 不使用')\n"
B1 = A1 + "    parser.add_argument('--noise_aug', default=0.0, type=float, help='[雜訊增強] 乘性斑點雜訊 ±比例（p=0.5）；0 = 不使用')\n"
A2 = "        transforms.RandomGamma(p=0.5)] if config['intensity_aug'] > 0 else []) + [\n"
B2 = ("        transforms.RandomGamma(p=0.5)] if config['intensity_aug'] > 0 else []) + ([   # [雜訊增強]\n"
      "        transforms.MultiplicativeNoise(multiplier=(1 - config['noise_aug'], 1 + config['noise_aug']), per_channel=False,\n"
      "                                       elementwise=True, p=0.5)] if config['noise_aug'] > 0 else []) + [\n")
for a, name in [(A1, '--intensity_aug 參數'), (A2, '[強度增強] train_transform')]:
    n = s.count(a)
    if n != 1:
        sys.exit(f'train.py 中「{name}」的程式碼出現 {n} 次（應為 1 次），未修改')
open(F, 'w', encoding='utf-8').write(s.replace(A1, B1).replace(A2, B2))
print('train.py 已加入 --noise_aug（[雜訊增強]）')