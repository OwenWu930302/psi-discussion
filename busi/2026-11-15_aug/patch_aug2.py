"""
在已含 [平移增強] 的 U-KAN train.py 再加入：--scale_aug、--rotate_aug（[平移增強] 同一步）、--intensity_aug（[強度增強]）
  全部預設 0：不給時 train_transform 與修改前完全相同（原版 / 只給 --shift_aug 的模型都不受影響）
  --scale_aug s   ：縮放 ±s（0.1 = ±10%）      --rotate_aug r：旋轉 ±r 度
  --intensity_aug a：RandomBrightnessContrast(亮度 ±a、對比 ±a, p=0.5) + RandomGamma(80–120, p=0.5)
可重複執行（已修改過會略過）；找不到預期的程式碼時不修改並報錯
用法（Seg_UKAN 資料夾）：python patch_aug2.py
"""
import sys

F = 'train.py'
s = open(F, encoding='utf-8').read()
if '[強度增強]' in s:
    print('train.py 已含 [強度增強]，略過')
    sys.exit()
if '[平移增強]' not in s:
    sys.exit('train.py 尚未執行 patch_shiftaug.py，未修改')

A1 = "    parser.add_argument('--shift_p', default=0.5, type=float, help='套用隨機平移的機率')\n"
B1 = A1 + ("    parser.add_argument('--scale_aug', default=0.0, type=float, help='[平移增強] 隨機縮放 ±比例；0 = 不使用')\n"
           "    parser.add_argument('--rotate_aug', default=0.0, type=float, help='[平移增強] 隨機旋轉 ±度；0 = 不使用')\n"
           "    parser.add_argument('--intensity_aug', default=0.0, type=float, help='[強度增強] 亮度/對比 ±比例 + gamma；0 = 不使用')\n")
A2 = """    ] + ([geometric.transforms.ShiftScaleRotate(shift_limit=config['shift_aug'], scale_limit=0, rotate_limit=0,   # [平移增強]
                                                border_mode=0, value=0, mask_value=0, p=config['shift_p'])]
         if config['shift_aug'] > 0 else []) + [
"""
B2 = """    ] + ([geometric.transforms.ShiftScaleRotate(shift_limit=config['shift_aug'], scale_limit=config['scale_aug'] or 0,   # [平移增強]（or 0：預設時與修改前完全相同）
                                                rotate_limit=config['rotate_aug'] or 0,
                                                border_mode=0, value=0, mask_value=0, p=config['shift_p'])]
         if config['shift_aug'] > 0 or config['scale_aug'] > 0 or config['rotate_aug'] > 0 else []) + ([   # [強度增強]
        transforms.RandomBrightnessContrast(brightness_limit=config['intensity_aug'], contrast_limit=config['intensity_aug'], p=0.5),
        transforms.RandomGamma(p=0.5)] if config['intensity_aug'] > 0 else []) + [
"""
for a, name in [(A1, '--shift_p 參數'), (A2, '[平移增強] train_transform')]:
    n = s.count(a)
    if n != 1:
        sys.exit(f'train.py 中「{name}」的程式碼出現 {n} 次（應為 1 次），未修改')
open(F, 'w', encoding='utf-8').write(s.replace(A1, B1).replace(A2, B2))
print('train.py 已加入 --scale_aug、--rotate_aug、--intensity_aug')