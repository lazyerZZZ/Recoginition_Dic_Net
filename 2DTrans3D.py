import cv2
import numpy as np
import os

# --- 5. 点云转换逻辑（全参数注入版） ---

# 1. 左相机内参矩阵 M1 (根据图片 Calibration_1)
fx1, fy1 = 9426.84, 9405.09
cx1, cy1 = 1023.67, 1050.64
M1 = np.array([
    [fx1,   0, cx1],
    [  0, fy1, cy1],
    [  0,   0,   1]
], dtype=np.float64)

# 左相机畸变系数
D1 = np.array([-0.118265, 22.0724, 0.0115377, 0.00425971, 0], dtype=np.float64)

# 2. 右相机内参矩阵 M2
fx2, fy2 = 9620.13, 9370.52
cx2, cy2 = 1099.7, 1122.3
M2 = np.array([
    [fx2,   0, cx2],
    [  0, fy2, cy2],
    [  0,   0,   1]
], dtype=np.float64)

# 右相机畸变系数
D2 = np.array([0.426466, 22.5269, 0.0212542, 0.0504823, 0], dtype=np.float64)

# 3. 旋转矩阵 R (将角度转为弧度，并构建旋转矩阵)
# 图片中：X: 1.22243deg, Y: -33.8813deg, Z: 1.22943deg
r_x = np.radians(1.22243)
r_y = np.radians(-33.8813)
r_z = np.radians(1.22943)

# 依次计算绕 X, Y, Z 轴的旋转矩阵
R_x = np.array([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)], [0, np.sin(r_x), np.cos(r_x)]])
R_y = np.array([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0], [-np.sin(r_y), 0, np.cos(r_y)]])
R_z = np.array([[np.cos(r_z), -np.sin(r_z), 0], [np.sin(r_z), np.cos(r_z), 0], [0, 0, 1]])
R = R_z @ R_y @ R_x  # 组合旋转矩阵

# 4. 平移向量 T (单位: mm)
T = np.array([[183.001], [6.1305], [50.3829]], dtype=np.float64)

# 5. 立体校正：计算重投影矩阵 Q
# 因为你的虚拟双目非对称性很强（绕Y轴转了33度），必须通过这个函数计算出理想的对齐矩阵
image_size = (2048, 2048) # 假设你的图像尺寸，根据实际修改
R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(
    M1, D1, M2, D2, image_size, R, T
)

# 6. 利用 Q 矩阵将视差图转换为 3D 点云
# 注：disparity 是之前通过 stereo.compute 算出来的
disparity = np.load(os.path.join('/home/wenhao/bishe_code/2DTrans3D_result', "disparity_tvl1.npy"))
disparity = disparity.astype(np.float32)
mask = disparity > 0
h, w = disparity.shape
u, v = np.meshgrid(np.arange(w), np.arange(h))

# 为每个像素构建 (u, v, d) 向量
points_2d = np.stack((u, v, disparity), axis=-1).reshape(-1, 3)

# 重投影到 3D 空间
# cv2.reprojectImageTo3D 内部使用的是标准的齐次坐标变换
points_3d = cv2.reprojectImageTo3D(disparity, Q)

z_channel = points_3d[:, :, 2]

# 过滤掉无效视差的点
point_cloud = points_3d[mask]

# 7. 保存点云为 .xyz 格式
save_path = os.path.join('/home/wenhao/bishe_code/2DTrans3D_result', "cloud_output_perfect.xyz")
np.savetxt(save_path, point_cloud, fmt='%.4f')

print(f"点云转换成功（已结合左右相机参数与立体校正）！")
print(f"有效生成点数: {point_cloud.shape[0]}")