"""
低分影像的疊合圖：原圖（256×256）+ 綠 = 標註、紅 = 各模型的預測（outputs/<模型>/pred_last/val/<影像>.png），黃字 = Dice
用法（Seg_UKAN 資料夾）：python overlay_low.py --cases "benign (299),malignant (116),benign (131)"
"""
import argparse
import os

import cv2
import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument('--cases', required=True)
ap.add_argument('--models', default='base2,base3,shift,shift_r2,ssr,drop1,drop3,speckle')
ap.add_argument('--labels', default='base2,base3,shift,shift_r2,ssr,drop10,drop30,speckle')
ap.add_argument('--output_dir', default='outputs')
ap.add_argument('--out', default='outputs/analysis_busi_overlay')
args = ap.parse_args()
models, labels = args.models.split(','), args.labels.split(',')
cfg = yaml.load(open(f'{args.output_dir}/busi_UKAN_base2/config.yml'), Loader=yaml.FullLoader)
base = os.path.join(cfg['data_dir'], cfg['dataset'])
H, W = cfg['input_h'], cfg['input_w']
os.makedirs(args.out, exist_ok=True)


def tile(img, gt, pr, text):
    o = img.copy()
    for m, col in [(gt, (0, 255, 0)), (pr, (0, 0, 255))]:
        if m is not None:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, col, 2)
    cv2.rectangle(o, (0, 0), (W, 22), (0, 0, 0), -1)
    cv2.putText(o, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return o


for c in args.cases.split(','):
    img = cv2.resize(cv2.imread(os.path.join(base, 'images', c + '.png')), (W, H), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(cv2.imread(os.path.join(base, 'masks', '0', c + '_mask.png'), cv2.IMREAD_GRAYSCALE), (W, H),
                    interpolation=cv2.INTER_NEAREST) > 127
    tiles = [tile(img, None, None, 'image'), tile(img, gt, None, 'ground truth')]
    for m, lab in zip(models, labels):
        f = f'{args.output_dir}/busi_UKAN_{m}/pred_last/val/{c}.png'
        if not os.path.exists(f):
            print(f'略過（找不到）：{f}')
            continue
        pr = cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127
        d = 2 * (pr & gt).sum() / max(pr.sum() + gt.sum(), 1)
        tiles.append(tile(img, gt, pr, f'{lab}  Dice {d:.2f}'))
    while len(tiles) % 5:
        tiles.append(np.zeros_like(img))
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    cv2.imwrite(f'{args.out}/{c}.png', np.vstack(rows))
    print(f'→ {args.out}/{c}.png')