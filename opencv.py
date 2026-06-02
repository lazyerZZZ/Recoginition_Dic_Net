import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# --- 1. 读取并预处理 ---
path_left = '/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000001_00.bmp'
path_right = '/home/wenhao/bishe_code/2DTrans3D_photoes/Camera00_00000002_00.bmp'

img_l = cv2.imread(path_left, 0)
img_r = cv2.imread(path_right, 0)

# 关键：稍微模糊一下，消除电子噪声对亚像素的干扰
img_l = cv2.GaussianBlur(img_l, (3, 3), 0)
img_r = cv2.GaussianBlur(img_r, (3, 3), 0)

h, w = img_l.shape
subset_size = 51  # 增大子集，提高鲁棒性
step = 20
# 搜索范围：假设右移在 50 到 150 像素之间（根据你的 cx2-cx1 偏移量调整）
search_range_x = [50, 150]

rows = range(subset_size, h - subset_size, step)
cols = range(subset_size, w - 160, step)
disp_map = np.zeros((len(rows), len(cols)))

print("开始执行鲁棒版局部匹配...")

for i, r in enumerate(rows):
    for j, c in enumerate(cols):
        # 提取左图特征块
        templ = img_l[r:r + subset_size, c:c + subset_size]

        # 动态定义右图搜索带 (只在水平方向搜索)
        search_min_x = max(0, c + search_range_x[0] - 20)
        search_max_x = min(w, c + search_range_x[1] + 20)
        search_area = img_r[r:r + subset_size, search_min_x:search_max_x]

        # 互相关匹配
        res = cv2.matchTemplate(search_area, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # 剔除低相关性的点（误匹配）
        if max_val < 0.5:
            disp_map[i, j] = np.nan
        else:
            # 这里的计算逻辑要非常小心
            # 实际视差 = (右图匹配点x) - (左图原点x)
            actual_x_in_right = search_min_x + max_loc[0]
            disp_map[i, j] = actual_x_in_right - c

# --- 2. 扣除系统偏置 ---
# 虚拟双目 offset = cx2 - cx1
offset = 1099.7 - 1023.67
final_disp = disp_map - offset

# --- 3. 结果验证与保存 ---
save_path = "/home/wenhao/bishe_code/2DTrans3D_result/final_dic_check.png"
plt.figure(figsize=(10, 8))
# 使用 vmin 和 vmax 锁定显示范围，防止杂点干扰颜色
plt.imshow(final_disp, cmap='jet', vmin=np.nanpercentile(final_disp, 5), vmax=np.nanpercentile(final_disp, 95))
plt.colorbar(label='Relative Disparity')
plt.savefig(save_path)
print(f"计算完成。中值视差为: {np.nanmedian(final_disp):.2f}")
