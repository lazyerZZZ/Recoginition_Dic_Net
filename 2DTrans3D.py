import cv2
import numpy as np

# --- 1. 参数注入 (来自你的标定图片 Calibration_1) ---
# 左光路虚拟相机参数
M1 = np.array([[9426.84, 0, 1023.67], [0, 9405.09, 1050.64], [0, 0, 1]], dtype=np.float64)
D1 = np.array([-0.118265, 22.0724, 0.0115377, 0.00425971, 0], dtype=np.float64)

# 右光路虚拟相机参数
M2 = np.array([[9620.13, 0, 1099.7], [0, 9370.52, 1122.3], [0, 0, 1]], dtype=np.float64)
D2 = np.array([0.426466, 22.5269, 0.0212542, 0.0504823, 0], dtype=np.float64)

# 旋转 R 与 平移 T
R_angles = np.radians([1.22243, -33.8813, 1.22943]) # X, Y, Z
# 构建旋转矩阵
Rx = np.array([[1,0,0], [0,np.cos(R_angles[0]),-np.sin(R_angles[0])], [0,np.sin(R_angles[0]),np.cos(R_angles[0])]])
Ry = np.array([[np.cos(R_angles[1]),0,np.sin(R_angles[1])], [0,1,0], [-np.sin(R_angles[1]),0,np.cos(R_angles[1])]])
Rz = np.array([[np.cos(R_angles[2]),-np.sin(R_angles[2]),0], [np.sin(R_angles[2]),np.cos(R_angles[2]),0], [0,0,1]])
R = Rz @ Ry @ Rx
T = np.array([[183.001], [6.1305], [50.3829]], dtype=np.float64)

# --- 2. 执行立体校正 (核心步骤) ---
# 这步会根据左右光路的差异，计算出一个让图像对齐的重投影矩阵 Q
img_size = (2048, 2048)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(M1, D1, M2, D2, img_size, R, T)

# --- 3. 转换点云 ---
# 使用平滑后的视差图
disparity = np.load('/home/wenhao/bishe_code/xyz/disparity_data_smoothed.npy').astype(np.float32)

# 注意：如果你的 SGBM 是在原始图像上做的，你需要减去主点偏移 cx2-cx1
# 因为 Q 矩阵通常预期视差是基于校正对齐后的图像
d_offset = M2[0,2] - M1[0,2]
disparity_corrected = disparity - d_offset

# 过滤掉噪点
mask = (disparity_corrected > 0) & (disparity_corrected < 200)

# 重投影
points_3d = cv2.reprojectImageTo3D(disparity_corrected, Q)
point_cloud = points_3d[mask]

# --- 4. 保存 ---
np.savetxt("/home/wenhao/bishe_code/xyz/cloud_output_final.xyz", point_cloud, fmt='%.4f')
print("点云计算完成，已综合左右光路全参数。")