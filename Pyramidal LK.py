import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import griddata

# --- 1. 读取图像 ---
path_left = '/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000001_00.bmp'
path_right = '/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000002_00.bmp'

img_l = cv2.imread(path_left, 0)
img_r = cv2.imread(path_right, 0)

# 轻微高斯模糊，抵抗散斑的高频散粒噪声
img_l = cv2.GaussianBlur(img_l, (3, 3), 0)
img_r = cv2.GaussianBlur(img_r, (3, 3), 0)

h, w = img_l.shape

print("正在构建计算网格...")
# --- 2. 在左图生成均匀的追踪网格 (ROI区域) ---
# 避开边缘，每隔 15 个像素取一个追踪点
margin = 50
step = 15
grid_x, grid_y = np.meshgrid(np.arange(margin, w - margin, step),
                             np.arange(margin, h - margin, step))
p0 = np.float32(np.vstack((grid_x.flatten(), grid_y.flatten())).T)
p0 = p0.reshape(-1, 1, 2)

# --- 3. 金字塔 Lucas-Kanade 亚像素光流 ---
print("正在进行金字塔光流追踪 (Coarse-to-Fine)...")
# 参数极其关键：
# winSize: 窗口大小，设为 41 保证包含足够的散斑特征
# maxLevel: 金字塔层数，设为 4 代表会缩小 2^4 = 16 倍进行粗匹配，彻底解决大位移脱靶
lk_params = dict(winSize=(41, 41),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001))

p1, st, err = cv2.calcOpticalFlowPyrLK(img_l, img_r, p0, None, **lk_params)

# --- 4. 提取有效点并计算水平视差 ---
# st == 1 代表追踪成功的点
good_old = p0[st == 1]
good_new = p1[st == 1]

# 计算水平位移 (X方向)
disparity_x = good_new[:, 0] - good_old[:, 0]

# --- 5. 扣除系统偏置 ---
offset = 1099.7 - 1023.67
aligned_disparity_x = disparity_x - offset

print(f"追踪成功点数: {len(aligned_disparity_x)}")
print(f"有效水平视差中值: {np.median(aligned_disparity_x):.4f}")

# --- 6. 插值恢复为稠密视差图 ---
print("正在插值生成稠密视差场...")
grid_x_dense, grid_y_dense = np.meshgrid(np.arange(0, w), np.arange(0, h))
dense_disparity = griddata((good_old[:, 0], good_old[:, 1]), aligned_disparity_x,
                           (grid_x_dense, grid_y_dense), method='linear')

# 处理边缘的 NaN 值
dense_disparity = np.nan_to_num(dense_disparity, nan=np.median(aligned_disparity_x))

# --- 7. 保存结果 ---
save_dir = "/home/wenhao/bishe_code/2DTrans3D_result"
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'disparity_pyrlk.npy'), dense_disparity)

plt.figure(figsize=(10, 8))
# 限制显示范围，排除极个别追踪失败的离群点干扰
vmin = np.percentile(aligned_disparity_x, 5)
vmax = np.percentile(aligned_disparity_x, 95)
plt.imshow(dense_disparity, cmap='jet', vmin=vmin, vmax=vmax)
plt.colorbar(label='Aligned Horizontal Disparity (pixels)')
plt.title("Pyramidal LK Sub-pixel Disparity Field")
plt.savefig(os.path.join(save_dir, 'pyrlk_disparity.png'))
print("计算完成，结果已保存！")