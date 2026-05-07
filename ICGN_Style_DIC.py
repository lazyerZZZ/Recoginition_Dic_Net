import numpy as np
import cv2
import os
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# --- 1. 参数配置 ---
SUBSET_SIZE = 41  # 子集大小 (类似 IC-GN 的 Window Size)
STEP_SIZE = 20  # 步长 (控制点云密度)
UPSAMPLE_FACTOR = 100  # 亚像素精度倍数 (100代表 0.01 像素精度)

path_left = '/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000001_00.bmp'
path_right = '/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000002_00.bmp'

img_l = cv2.imread(path_left, 0)
img_r = cv2.imread(path_right, 0)

h, w = img_l.shape
# 定义计算区域 (避开边缘)
rows = range(SUBSET_SIZE, h - SUBSET_SIZE, STEP_SIZE)
cols = range(SUBSET_SIZE, w - SUBSET_SIZE, STEP_SIZE)

# 存储结果
disparity_map = np.zeros((len(rows), len(cols)))

print(f"开始 IC-GN 风格亚像素计算，预计计算点数: {len(rows) * len(cols)}...")

# --- 2. 局部亚像素迭代搜索 ---
for i, r in enumerate(rows):
    for j, c in enumerate(cols):
        # 提取左图子集
        subset_l = img_l[r:r + SUBSET_SIZE, c:c + SUBSET_SIZE]

        # 定义右图搜索区域 (假设水平位移在 0-160 之间)
        search_r = img_r[r:r + SUBSET_SIZE, max(0, c - 20):min(w, c + 180)]

        try:
            # phase_cross_correlation 提供亚像素级偏移检测
            # 这种方法在频域计算，对散斑图极其精准
            shift, error, diffphase = phase_cross_correlation(
                subset_l, search_r,
                upsample_factor=UPSAMPLE_FACTOR,
                normalization=None
            )
            # 我们只需要水平位移 (shift[1])，并修正搜索框起始位置
            disparity_map[i, j] = shift[1] - 20
        except:
            disparity_map[i, j] = np.nan

# --- 3. 后处理与对齐 ---
# 自动扣除系统偏置 cx2 - cx1
offset = 1099.7 - 1023.67
disparity_final = -(disparity_map + offset)  # 取反以符合深度公式

# --- 4. 保存与可视化 ---
save_dir = "/home/wenhao/bishe_code/2DTrans3D_result"
os.makedirs(save_dir, exist_ok=True)

# 保存数据
np.save(os.path.join(save_dir, 'disparity_icgn.npy'), disparity_final)

# 画图
plt.figure(figsize=(10, 8))
plt.imshow(disparity_final, cmap='jet', interpolation='bilinear')
plt.colorbar(label='Sub-pixel Disparity')
plt.title("IC-GN Style Local Sub-pixel Disparity Map")
plt.savefig(os.path.join(save_dir, 'icgn_disparity_result.png'))

print(f"计算完成！结果已存入 {save_dir}")