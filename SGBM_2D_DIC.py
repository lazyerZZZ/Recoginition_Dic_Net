import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 路径设置
base_path = '/home/wenhao/bishe_code/2DTrans3D_photoes/'
path_left = os.path.join(base_path, 'Camera00_00000001_00.bmp')
path_right = os.path.join(base_path, 'Camera00_00000002_00.bmp')

img_left = cv2.imread(path_left, cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread(path_right, cv2.IMREAD_GRAYSCALE)

if img_left is None or img_right is None:
    print("错误：无法读取图片")
    exit()

# 2. 配置 SGBM 参数
window_size = 9
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=160,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# --- 3. 计算 U 场 (水平方向) ---
print("正在计算水平视差 U...")
disparity_u = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
u_field = disparity_u - (1099.7 - 1023.67) # 补偿主点偏移

# --- 4. 计算 V 场 (竖直方向 - 采用旋转技巧) ---
print("正在通过旋转计算竖直视差 V...")
# 顺时针旋转 90 度 (cv2.ROTATE_90_CLOCKWISE)
img_left_rot = cv2.rotate(img_left, cv2.ROTATE_90_CLOCKWISE)
img_right_rot = cv2.rotate(img_right, cv2.ROTATE_90_CLOCKWISE)

# 在旋转后的图上运行 SGBM
disparity_v_rot = stereo.compute(img_left_rot, img_right_rot).astype(np.float32) / 16.0

# 逆时针旋转回来 (cv2.ROTATE_90_COUNTERCLOCKWISE)
v_field = cv2.rotate(disparity_v_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)

# --- 5. 绘图并保存 PNG ---
target_dir = "/home/wenhao/bishe_code/2DTrans3D_result"
os.makedirs(target_dir, exist_ok=True)

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title('U Field (Horizontal via SGBM)')
plt.imshow(u_field, cmap='jet')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('V Field (Vertical via Rotated SGBM)')
plt.imshow(v_field, cmap='jet')
plt.colorbar()
plt.axis('off')

output_path = os.path.join(target_dir, 'sgbm_uv_dual_direction.png')
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print("-" * 30)
print(f"完成！已通过旋转法强行让 SGBM 计算了竖直方向。")
print(f"结果图片: {output_path}")
print("-" * 30)