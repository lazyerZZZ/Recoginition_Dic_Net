import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# --- 1. 参数手动校准 (基于你肉眼观察的“右移”) ---
SUBSET_SIZE = 41  # 子集大小
STEP_SIZE = 15  # 步长，越小点越密
SEARCH_MARGIN = 60  # 在右图搜索时，在左图位置的基础上额外向右看 60 像素

path_left = '/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000001_00.bmp'
path_right = '/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000002_00.bmp'

img_l = cv2.imread(path_left, 0)
img_r = cv2.imread(path_right, 0)

# 预处理：增强对比度，让散斑更跳跃
img_l = cv2.equalizeHist(img_l)
img_r = cv2.equalizeHist(img_r)

h, w = img_l.shape
rows = range(SUBSET_SIZE, h - SUBSET_SIZE, STEP_SIZE)
cols = range(SUBSET_SIZE, w - SUBSET_SIZE - SEARCH_MARGIN, STEP_SIZE)
disparity_map = np.zeros((len(rows), len(cols)))

print("正在进行空间域子集匹配...")

# --- 2. 空间域迭代搜索 ---
for i, r in enumerate(rows):
    for j, c in enumerate(cols):
        # 左图模板
        template = img_l[r:r + SUBSET_SIZE, c:c + SUBSET_SIZE]

        # 右图搜索区域：根据“右移”观察，向右多扩一点
        # 搜索范围设为 [c-20, c+SEARCH_MARGIN]
        search_area = img_r[r:r + SUBSET_SIZE, c - 10:c + SEARCH_MARGIN]

        # 归一化相关系数匹配 (最接近 DIC 原理)
        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # max_loc[0] 就是在 search_area 里的相对水平位移
        # 亚像素修正：通过周围像素进行二次拟合（简易版 IC-GN 逻辑）
        raw_x = max_loc[0]
        if 0 < raw_x < res.shape[1] - 1:
            # 抛物线插值实现亚像素精度
            y0, y1, y2 = res[0, raw_x - 1], res[0, raw_x], res[0, raw_x + 1]
            subpixel_x = raw_x + (y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-5)
        else:
            subpixel_x = raw_x

        disparity_map[i, j] = subpixel_x - 10  # 减去 search_area 的起始偏移

# --- 3. 物理对齐与保存 ---
offset = 1099.7 - 1023.67
# 最终视差 = 计算视差 - 系统偏置
disparity_final = disparity_map - offset

save_dir = "/home/wenhao/bishe_code/2DTrans3D_result"
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'disparity_final_v2.npy'), disparity_final)

# 画图：如果这次还是全黑，请检查控制台打印的 disparity_final.max()
plt.figure(figsize=(10, 8))
plt.imshow(disparity_final, cmap='jet')
plt.colorbar(label='Disparity')
plt.savefig(os.path.join(save_dir, 'dic_v2_result.png'))
print(f"计算完成！中值视差: {np.median(disparity_final):.4f}")
