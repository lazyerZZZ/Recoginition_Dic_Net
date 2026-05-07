import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import griddata

# --- 1. 初始化路径 ---
save_dir = "/home/wenhao/bishe_code/2DTrans3D_result/"
os.makedirs(save_dir, exist_ok=True)

path_left = '/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000001_00.bmp'
path_right = '/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000002_00.bmp'

img_l = cv2.imread(path_left, 0)
img_r = cv2.imread(path_right, 0)
h, w = img_l.shape # 2048, 2048

# --- 2. 提取特征与大位移匹配 ---
print("正在提取 SIFT 特征并进行全图匹配...")
sift = cv2.SIFT_create(nfeatures=20000)
kp1, des1 = sift.detectAndCompute(img_l, None)
kp2, des2 = sift.detectAndCompute(img_r, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.7 * n.distance]

pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# RANSAC 物理提纯，确保位移矢量符合几何逻辑
_, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
v_pts1 = pts1[mask.ravel() == 1]
v_pts2 = pts2[mask.ravel() == 1]

# 原始位移量 (不扣除 offset)
raw_u = v_pts2[:, 0] - v_pts1[:, 0]
raw_v = v_pts2[:, 1] - v_pts1[:, 1]

# --- 3. 稠密化：恢复到 2048x2048 坐标系 ---
print(f"正在插值生成 {h}x{w} 稠密视差场...")
# 生成与原图完全一致的坐标网格
grid_y, grid_x = np.mgrid[0:h:1, 0:w:1]

# 使用线性插值将稀疏位移扩充到全图像素点
dense_u = griddata(v_pts1, raw_u, (grid_x, grid_y), method='linear')
dense_v = griddata(v_pts1, raw_v, (grid_x, grid_y), method='linear')

# 填充边缘 NaN 值为全局中值，防止后续 3D 转换报错
dense_u = np.nan_to_num(dense_u, nan=np.nanmedian(raw_u))
dense_v = np.nan_to_num(dense_v, nan=np.nanmedian(raw_v))

# --- 4. 存储数据 ---
np.save(os.path.join(save_dir, 'raw_disp_u.npy'), dense_u)
np.save(os.path.join(save_dir, 'raw_disp_v.npy'), dense_v)

# --- 5. 可视化诊断 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# 水平位移图 (U)
im1 = ax1.imshow(dense_u, cmap='jet')
ax1.set_title(f"U Field (Raw)\nMedian: {np.nanmedian(raw_u):.2f} px")
fig.colorbar(im1, ax=ax1)

# 垂直位移图 (V)
im2 = ax2.imshow(dense_v, cmap='jet')
ax2.set_title(f"V Field (Raw)\nMedian: {np.nanmedian(raw_v):.2f} px")
fig.colorbar(im2, ax=ax2)

plt.savefig(os.path.join(save_dir, 'uv_displacement_check.png'), dpi=300)

print("-" * 30)
print(f"计算成功！结果已存入: {save_dir}")
print(f"U 场中值: {np.nanmedian(raw_u):.2f} (这应该接近 100)")
print(f"V 场中值: {np.nanmedian(raw_v):.2f} (这应该接近 0)")
print("-" * 30)