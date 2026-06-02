import numpy as np
import cv2
import os
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

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
margin = 50
step = 15
grid_x, grid_y = np.meshgrid(np.arange(margin, w - margin, step),
                             np.arange(margin, h - margin, step))
p0 = np.float32(np.vstack((grid_x.flatten(), grid_y.flatten())).T)
p0 = p0.reshape(-1, 1, 2)

# --- 3. 金字塔 Lucas-Kanade 亚像素光流 ---
print("正在进行金字塔光流追踪 (Coarse-to-Fine)...")
lk_params = {
    'winSize': (41, 41),
    'maxLevel': 4,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001),
}

p1, st, err = cv2.calcOpticalFlowPyrLK(img_l, img_r, p0, None, **lk_params)

# --- 4. 提取有效点并计算水平/垂直视差 ---
good_old = p0[st == 1]
good_new = p1[st == 1]

# 计算水平位移 (X方向)
disparity_x = good_new[:, 0] - good_old[:, 0]
# 新增：计算垂直位移 (Y方向)
disparity_y = good_new[:, 1] - good_old[:, 1]

# --- 5. 扣除系统偏置 ---
offset = 1099.7 - 1023.67
aligned_disparity_x = disparity_x - offset
# 垂直方向通常不需要偏置，直接使用
aligned_disparity_y = disparity_y

print(f"追踪成功点数: {len(aligned_disparity_x)}")

# --- 6. 插值恢复为稠密视差图 ---
print("正在插值生成稠密视差场...")
grid_x_dense, grid_y_dense = np.meshgrid(np.arange(0, w), np.arange(0, h))

# 水平方向插值 (保持原逻辑)
dense_disparity_x = griddata((good_old[:, 0], good_old[:, 1]), aligned_disparity_x,
                             (grid_x_dense, grid_y_dense), method='linear')
dense_disparity_x = np.nan_to_num(dense_disparity_x, nan=np.median(aligned_disparity_x))

# 新增：垂直方向插值 (完全对齐原逻辑)
dense_disparity_y = griddata((good_old[:, 0], good_old[:, 1]), aligned_disparity_y,
                             (grid_x_dense, grid_y_dense), method='linear')
dense_disparity_y = np.nan_to_num(dense_disparity_y, nan=np.median(aligned_disparity_y))

# --- 7. 保存结果 ---
save_dir = "/home/wenhao/bishe_code/2DTrans3D_result"
os.makedirs(save_dir, exist_ok=True)

# 保存数据
np.save(os.path.join(save_dir, 'u_field_pyrlk.npy'), dense_disparity_x)
np.save(os.path.join(save_dir, 'v_field_pyrlk.npy'), dense_disparity_y)

# 绘图 1: 水平视差 (保持原渲染逻辑)
plt.figure(figsize=(10, 8))
vmin_x = np.percentile(aligned_disparity_x, 5)
vmax_x = np.percentile(aligned_disparity_x, 95)
plt.imshow(dense_disparity_x, cmap='jet', vmin=vmin_x, vmax=vmax_x)
plt.colorbar(label='Aligned Horizontal Disparity (pixels)')
plt.title("U Field (Horizontal Disparity)")
plt.savefig(os.path.join(save_dir, 'pyrlk_disparity_U.png'))
plt.close()

# 绘图 2: 垂直视差 (新增输出)
plt.figure(figsize=(10, 8))
vmin_y = np.percentile(aligned_disparity_y, 5)
vmax_y = np.percentile(aligned_disparity_y, 95)
plt.imshow(dense_disparity_y, cmap='jet', vmin=vmin_y, vmax=vmax_y)
plt.colorbar(label='Vertical Disparity (pixels)')
plt.title("V Field (Vertical Disparity)")
plt.savefig(os.path.join(save_dir, 'pyrlk_disparity_V.png'))
plt.close()

print("计算完成，U/V 两个维度的结果已保存！")
