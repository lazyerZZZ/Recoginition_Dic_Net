import cv2
import numpy as np
import os
from skimage.registration import optical_flow_tvl1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# 1. 加载图像并预处理
img_left = cv2.imread('/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000001_00.bmp', 0)
img_right = cv2.imread('/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000002_00.bmp', 0)

if img_left is None or img_right is None:
    print("错误：无法读取图像，请检查路径。")
    exit()

# 2. 使用 TV-L1 光流法计算
# 返回值说明：v 是垂直位移，u 是水平位移
print("正在计算 TV-L1 全场位移 (可能需要一些时间)...")
v, u = optical_flow_tvl1(img_left, img_right,
                         attachment=15,
                         tightness=0.3,
                         num_warp=5,
                         num_iter=10)

# 3. 处理位移场
# U 场（水平）：扣除系统偏置 (cx2 - cx1)
offset = 1099.7 - 1023.67
u_final = u - offset

# V 场（垂直）：通常不需要偏置补偿
v_final = v

# 4. 确保输出目录存在
save_dir = "/home/wenhao/bishe_code/2DTrans3D_result"
os.makedirs(save_dir, exist_ok=True)

# 定义一个绘图函数减少重复代码


def save_disparity_map(data, title, filename):
    plt.figure(figsize=(12, 9))
    # 剪掉边缘噪声（光流法边缘通常不稳定）
    crop_size = 160
    plt.imshow(data[:, crop_size:], cmap='jet')
    plt.colorbar(label='Pixel Shift')
    plt.title(title)
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    return path


# 5. 分别生成并保存两个视差场图片
u_path = save_disparity_map(u_final, "Horizontal Disparity (U Field) - TV-L1", "result_U_field.png")
v_path = save_disparity_map(v_final, "Vertical Disparity (V Field) - TV-L1", "result_V_field.png")

# 6. 保存 NPY 数据供后续 3D 转换使用
np.save(os.path.join(save_dir, 'u_field_tvl1.npy'), u_final)
np.save(os.path.join(save_dir, 'v_field_tvl1.npy'), v_final)

# --- 验证环节 ---
print("=" * 50)
print("成功！已生成两个视差场文件：")
print(f"1. 水平 U 场: {u_path}")
print(f"2. 竖直 V 场: {v_path}")
print("数据已导出至 .npy 文件中。")
print("=" * 50)
