import numpy as np
from scipy.interpolate import RectBivariateSpline
import os

# 1. 加载你原来有阶梯的视差图
target_dir = "/home/wenhao/bishe_code/xyz"
original_disp_path = os.path.join(target_dir, 'disparity_data.npy')
disparity = np.load(original_disp_path)

# 2. 检查是否有无效点 (SGBM生成的无效点通常是负数或无穷大)
valid_mask = (disparity > 0) & np.isfinite(disparity)
disparity[~valid_mask] = np.mean(disparity[valid_mask])  # 用均值简单填充无效点，方便插值

h, w = disparity.shape
x = np.arange(w)
y = np.arange(h)

# 3. 创建二维样条插值函数 (kx, ky 设为 3，即三次样条，能保证极高的平滑度)
# 注意：这一步在大图像上可能需要几分钟，请耐心等待
print("正在进行三次样条插值，这可能需要一些时间...")
spline_func = RectBivariateSpline(y, x, disparity, kx=3, ky=3)

# 4. 在同一个格点上重新采样 (或者你可以在更细的格点上采样，但目前没必要)
new_disparity = spline_func(y, x)

# 5. 恢复无效点标记
new_disparity[~valid_mask] = -1

# 6. 保存平滑后的新视差图
smoothed_disp_path = os.path.join(target_dir, 'disparity_data_smoothed.npy')
np.save(smoothed_disp_path, new_disparity)
print(f"平滑后的视差图已保存至: {smoothed_disp_path}")

# # 7. 可选：可视化对比并保存为图片
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(disparity, cmap='jet')
# plt.title('Original (With Steps)')
# plt.colorbar()
#
# plt.subplot(122)
# plt.imshow(new_disparity, cmap='jet')
# plt.title('Smoothed (DIC Accurate)')
# plt.colorbar()
#
# # 不要用 plt.show()，改用 savefig
# plt.savefig('/home/wenhao/bishe_code/comparison_result.png')
# print("对比图已保存至: comparison_result.png")
