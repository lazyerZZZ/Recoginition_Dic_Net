import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 确保使用绝对路径
base_path = '/home/wenhao/bishe_code/2DTrans3D_photoes/'
path_left = os.path.join(base_path, 'Camera00_00000001_00.bmp')
path_right = os.path.join(base_path, 'Camera00_00000002_00.bmp')

# 检查图片是否存在
if not os.path.exists(path_left):
    print(f"致命错误：找不到左图，请检查路径: {path_left}")
    exit()

img_left = cv2.imread(path_left, cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread(path_right, cv2.IMREAD_GRAYSCALE)

# 2. 计算视差 (保持之前的优化参数)
window_size = 9
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=160,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

# 3. 补偿主点偏移 (cx2 - cx1)
disparity_aligned = disparity - (1099.7 - 1023.67)

# --- 4. 关键：强制保存逻辑 ---
target_dir = "/home/wenhao/bishe_code/2DTrans3D_result"

# 如果没有文件夹就创建，如果有就跳过
os.makedirs(target_dir, exist_ok=True)

# 保存 NPY
npy_path = os.path.join(target_dir, 'disparity_aligned.npy')
np.save(npy_path, disparity_aligned)

# 保存图片
plt.figure(figsize=(10, 8))
plt.imshow(disparity_aligned, cmap='jet')
plt.colorbar()
png_path = os.path.join(target_dir, 'enhanced_disparity.png')
plt.savefig(png_path)
plt.close() # 释放内存

# 5. 打印确认信息 (如果看到这两行，文件一定生成了)
print("-" * 30)
print(f"成功！文件已保存到：{target_dir}")
print(f"文件1: {os.path.basename(npy_path)}")
print(f"文件2: {os.path.basename(png_path)}")
print("-" * 30)
