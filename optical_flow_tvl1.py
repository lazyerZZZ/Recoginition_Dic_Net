
import cv2
from skimage.registration import optical_flow_tvl1 # 使用全变分光流法，更平滑
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前调用
import matplotlib.pyplot as plt
import numpy as np
import os
# ... 其余 import

# 1. 加载图像并预处理
img_left = cv2.imread('/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000001_00.bmp', 0)
img_right = cv2.imread('/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000002_00.bmp', 0)

# 2. 使用 TV-L1 光流法（一种高级的连续位移场计算方法）
# 它比 SGBM 强在：它假设位移场是连续变化的，能有效抑制“多层平板”
# attachment控制数据保真度，tightness控制平滑度
v, u = optical_flow_tvl1(img_left, img_right,
                         attachment=15,
                         tightness=0.3,
                         num_warp=5,
                         num_iter=10)

# u 就是我们需要的水平视差场 (Horizontal Disparity)
disparity_advanced = u

# 3. 自动扣除系统偏置 (cx2 - cx1)
offset = 1099.7 - 1023.67
disparity_final = disparity_advanced - offset

# 确保路径是绝对路径，避免存错地方
save_dir = "/home/wenhao/bishe_code/2DTrans3D_result"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'advanced_dic_result.png')

# 绘图逻辑
plt.figure(figsize=(12, 9))
# 剪掉左侧无效边缘（160是numDisparities，不剪掉的话颜色会被边缘噪声带偏）
plt.imshow(disparity_final[:, 160:], cmap='jet')
plt.colorbar(label='Disparity (pixels)')
plt.title("Advanced TV-L1 Disparity Map")

# 强制保存
plt.savefig(save_path, dpi=300)
plt.close() # 必须关闭，否则内存占用会导致后续报错

# --- 验证环节 ---
if os.path.exists(save_path):
    print("=" * 50)
    print(f"成功！视差场照片已生成。")
    print(f"文件大小: {os.path.getsize(save_path) / 1024:.2f} KB")
    print(f"具体位置: {save_path}")
    print("=" * 50)
else:
    print("❌ 错误：图片保存失败，请检查目录权限或磁盘空间。")

# 保存用于 3D 转换
np.save('//home/wenhao/bishe_code/2DTrans3D_result/disparity_tvl1.npy', disparity_final)