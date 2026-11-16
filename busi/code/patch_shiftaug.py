"""
在 U-KAN train.py 加入 --shift_aug（隨機平移增強；[平移增強]）
  不給 --shift_aug（預設 0）時，train_transform 與原版完全相同
  給 --shift_aug s：在 Resize(256) 之後加 ShiftScaleRotate(只平移 ±s×256 px，縮放 0、旋轉 0，邊緣補 0，機率 --shift_p)
可重複執行（已修改過會略過）；找不到預期的程式碼時不修改並報錯
用法（Seg_UKAN 資料夾）：python patch_shiftaug.py
"""
import sys

F = 'train.py'
s = open(F, encoding='utf-8').read()
if '[平移增強]' in s:
    print('train.py 已含 [平移增強]，略過')
    sys.exit()

A1 = "    parser.add_argument('--move_to_train', default=None, help='移到訓練集的測試影像清單 csv（第一欄 case）')\n"
B1 = A1 + ("    # [平移增強] 不給 --shift_aug 時與原版完全相同\n"
           "    parser.add_argument('--shift_aug', default=0.0, type=float, help='隨機平移上限（影像寬高的比例，0.125 = 32 px）；0 = 不使用')\n"
           "    parser.add_argument('--shift_p', default=0.5, type=float, help='套用隨機平移的機率')\n")
A2 = """        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose(["""
B2 = """        Resize(config['input_h'], config['input_w']),
    ] + ([geometric.transforms.ShiftScaleRotate(shift_limit=config['shift_aug'], scale_limit=0, rotate_limit=0,   # [平移增強]
                                                border_mode=0, value=0, mask_value=0, p=config['shift_p'])]
         if config['shift_aug'] > 0 else []) + [
        transforms.Normalize(),
    ])

    val_transform = Compose(["""
for a, name in [(A1, '--move_to_train 參數'), (A2, 'train_transform')]:
    n = s.count(a)
    if n != 1:
        sys.exit(f'train.py 中「{name}」的程式碼出現 {n} 次（應為 1 次），未修改')
s = s.replace(A1, B1).replace(A2, B2)
open(F, 'w', encoding='utf-8').write(s)
print('train.py 已加入 --shift_aug、--shift_p（[平移增強]）')