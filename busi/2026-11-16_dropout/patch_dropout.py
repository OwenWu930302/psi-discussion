"""
U-KAN 加入 Dropout（[Dropout]）
  archs.py：KANLayer 原版建立了 self.drop = nn.Dropout(drop)，但 forward 從未使用 → drop_rate 原本沒有作用
            在 KANLayer 輸出加上 self.drop（只在 p > 0 時呼叫；p = 0 時運算與亂數完全與原版相同）
  train.py：加入 --drop_rate（KAN 層 Dropout 機率）、--drop_path_rate（原版已支援的 DropPath / stochastic depth 上限）
            兩者預設 0 = 原版（UKAN 建構子的預設值也是 0）
可重複執行（已修改過會略過）；找不到預期的程式碼時不修改並報錯
用法（Seg_UKAN 資料夾）：python patch_dropout.py
"""
import sys


def patch(F, pairs, tag):
    s = open(F, encoding='utf-8').read()
    if tag in s:
        print(f'{F} 已含 {tag}，略過')
        return
    for a, b in pairs:
        n = s.count(a)
        if n != 1:
            sys.exit(f'{F} 中預期的程式碼出現 {n} 次（應為 1 次），未修改：\n{a}')
    for a, b in pairs:
        s = s.replace(a, b)
    open(F, 'w', encoding='utf-8').write(s)
    print(f'{F} 已修改（{tag}）')


patch('archs.py', [(
    "        x = self.dwconv_3(x, H, W)\n",
    "        x = self.dwconv_3(x, H, W)\n"
    "        if self.drop.p > 0:   # [Dropout] 原版建立了 self.drop 但未使用；p = 0 時與原版完全相同\n"
    "            x = self.drop(x)\n")], '[Dropout]')

patch('train.py', [
    ("    parser.add_argument('--move_to_train', default=None,",
     "    parser.add_argument('--drop_rate', default=0.0, type=float, help='[Dropout] KAN 層 Dropout 機率；0 = 原版')\n"
     "    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='[Dropout] DropPath 上限；0 = 原版')\n"
     "    parser.add_argument('--move_to_train', default=None,"),
    ("embed_dims=config['input_list'], no_kan=config['no_kan'])",
     "embed_dims=config['input_list'], no_kan=config['no_kan'],\n"
     "                                           drop_rate=config['drop_rate'], drop_path_rate=config['drop_path_rate'])   # [Dropout]")],
    '[Dropout]')