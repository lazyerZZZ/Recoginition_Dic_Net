import cv2
import numpy as np
import os
import matplotlib
# 强制使用非交互式后端，避免 PyCharm 插件报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 1. 路径设置 ---
base_path = '/home/wenhao/bishe_code/2DTrans3D_photoes/'
path_left = os.path.join(base_path, 'Camera00_00000001_00.bmp')
path_right = os.path.join(base_path, 'Camera00_00000002_00.bmp')

# --- 2. 读取图像 ---
img_left = cv2.imread(path_left, cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread(path_right, cv2.IMREAD_GRAYSCALE)

if img_left is None:
    print("图像读取失败，请检查路径。")
    exit()

# --- 3. SGBM 视差计算 ---
window_size = 5
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # 你图像是2048x2048，这个范围可以设大一点
    blockSize=window_size,
    P1=8 * 1 * window_size**2,
    P2=32 * 1 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

# --- 4. 保存视差场图片（替代 plt.show()） ---
plt.figure(figsize=(10, 8))
plt.imshow(disparity, cmap='jet')
plt.colorbar(label='Disparity (pixels)')
plt.title('Disparity Map')
plt.savefig(os.path.join(base_path, 'disparity_result.png')) # 保存到本地文件夹
print("视差场结果图已保存至: disparity_result.png")

# --- 5. 点云转换逻辑 ---
# 请根据你的图三（相机参数图片）修改以下数值
fx = 2500.0  # 焦距像素值
fy = 2500.0
cx = 1024.0  # 图像中心 u
cy = 1024.0  # 图像中心 v
B = 50.0     # 双光路等效基线 (mm)

# 过滤无效点并计算
mask = disparity > 0
h, w = disparity.shape
u, v = np.meshgrid(np.arange(w), np.arange(h))

z = (fx * B) / disparity[mask]
x = (u[mask] - cx) * z / fx
y = (v[mask] - cy) * z / fy

point_cloud = np.vstack((x, y, z)).T

# 保存为 .xyz 格式，可以用 MeshLab 打开
np.savetxt(os.path.join('/home/wenhao/bishe_code', "cloud_output.xyz"), point_cloud, fmt='%.4f')
print(f"点云转换成功！生成点数: {point_cloud.shape[0]}")
print("点云文件已保存至: cloud_output.xyz")

# 将视差矩阵保存为 numpy 格式，方便随时加载
import numpy as np
np.save('/xyz/disparity_data.npy', disparity)
