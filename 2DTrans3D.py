import numpy as np
import cv2
import os
import open3d as o3d
from scipy.ndimage import median_filter

def get_rotation_matrix(x_deg, y_deg, z_deg):
    """将欧拉角转换为旋转矩阵"""
    # 转换弧度
    rx, ry, rz = np.radians([x_deg, y_deg, z_deg])
    # 计算旋转矩阵 (一般遵循 Z-Y-X 顺序)
    R_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    R_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return R_z @ R_y @ R_x


def reconstruct_3d_pro():
    # --- 1. 输入路径设置 ---
    data_dir = "/home/wenhao/bishe_code/2DTrans3D_result/"
    u_field = np.load(os.path.join(data_dir, 'raw_disp_u.npy'))
    v_field = np.load(os.path.join(data_dir, 'raw_disp_v.npy'))
    h, w = u_field.shape  # 2048x2048

    # ----------------- 附加步骤：方法3 - 视差场中值偏差过滤 -----------------
    print("正在对视差场进行中值偏差检测（剔除孤岛突变点）...")

    # 1. 设置滤波窗口大小 (例如 5x5 像素)
    window_size = 5

    # 2. 计算局部中值 (忽略 NaN，所以需要对 NaN 做一下特殊处理或直接使用 nanmedian)
    # 为了安全，我们用 scipy 配合 numpy 掩码处理
    u_median = median_filter(np.nan_to_num(u_field, nan=0.0), size=window_size)
    v_median = median_filter(np.nan_to_num(v_field, nan=0.0), size=window_size)

    # 3. 计算原始视差与周围邻域中值的绝对差异
    u_diff = np.abs(u_field - u_median)
    v_diff = np.abs(v_field - v_median)

    # 4. 设定突变跳变阈值（比如像素位移突然比周围大 0.5 个像素，就认为是错配的尖刺）
    jump_threshold = 0.5

    # 5. 找到尖刺位置的掩码
    spike_mask = (u_diff > jump_threshold) | (v_diff > jump_threshold)

    # 6. 把这些尖刺位置的视差强制设为 NaN，这样后续 mask_final 就会自动把它们抛弃
    u_field[spike_mask] = np.nan
    v_field[spike_mask] = np.nan

    print(f"已在视差场层面剔除 {np.sum(spike_mask)} 个突变像素点。")

    # ----------------- 附加步骤：方法4 - 双边滤波平滑视差场 -----------------
    print("正在对视差场进行双边滤波平滑...")

    # OpenCV 的滤波不支持 NaN，需要先记录 NaN 的位置并填充为 0
    nan_mask = np.isnan(u_field) | np.isnan(v_field)
    u_temp = np.nan_to_num(u_field, nan=0.0).astype(np.float32)
    v_temp = np.nan_to_num(v_field, nan=0.0).astype(np.float32)

    # 执行双边滤波
    # d: 像素邻域直径 (比如 9)
    # sigmaColor: 视差差值阈值，越小越能保护真实边界 (设为 0.5 像素)
    # sigmaSpace: 空间距离阈值，越大平滑效果越强 (设为 5-10)
    u_smooth = cv2.bilateralFilter(u_temp, d=9, sigmaColor=0.5, sigmaSpace=5)
    v_smooth = cv2.bilateralFilter(v_temp, d=9, sigmaColor=0.5, sigmaSpace=5)

    # 把原来是 NaN 的地方重新设回 NaN，保持掩码的正确性
    u_smooth[nan_mask] = np.nan
    v_smooth[nan_mask] = np.nan

    # 用平滑后的场覆盖原场
    u_field = u_smooth
    v_field = v_smooth

    # --- 2. 标定参数录入 ---
    # 左相机 K1
    K1 = np.array([[9426.84, -54.3315, 1023.97],
                   [0, 9405.09, 1050.64],
                   [0, 0, 1]])
    dist1 = np.array([-0.118265, 22.0724, 0.0115377, -0.00425971, 0])  # k1,k2,p1,p2,k3

    # 右相机 K2
    K2 = np.array([[9620.13, -124.281, 1099.7],
                   [0, 9370.52, 1122.3],
                   [0, 0, 1]])
    dist2 = np.array([0.426466, 22.5269, 0.0212542, 0.0504823, 0])

    # 外参 (右相对于左)
    R = get_rotation_matrix(1.22243, -33.8813, 1.22943)
    T = np.array([[183.001], [6.1305], [50.3829]])

    # --- 3. 构建投影矩阵 P = K [R|t] ---
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # 左相机作为原点
    P2 = K2 @ np.hstack((R, T))

    # --- 4. 准备像素坐标对 ---
    print("正在处理像素坐标...")
    y, x = np.mgrid[0:h, 0:w]
    pts1 = np.vstack((x.flatten(), y.flatten())).T.reshape(-1, 1, 2)

    # 根据视差场找到右图对应点
    x2 = x + u_field
    y2 = y + v_field
    pts2 = np.vstack((x2.flatten(), y2.flatten())).T.reshape(-1, 1, 2)

    # --- 5. 坐标去畸变 (关键：消除镜头扭曲) ---
    print("正在科学裁剪 ROI 安全区域...")
    mask_roi = np.zeros_like(u_field, dtype=bool)

    # 定义安全收缩边界 (根据你的实验图，建议收缩50像素)
    margin = 200
    h, w = u_field.shape

    # 将内部 2048 - 50*2 = 1948x1948 的区域设为有效
    mask_roi[margin:h - margin, margin:w - margin] = True

    # 3. 组合过滤：保留在 ROI 内且不是 NaN 的点 (如果有的话)
    # linear griddata 默认包围圈外是NaN
    # 我们不仅要 NaN，还要 ROI 内部的点
    # 如果你在 DIC 代码里用中值填充了 NaNs，请注意这里的处理
    mask_final = mask_roi & (~np.isnan(u_field)) & (~np.isnan(v_field))
    flat_mask = mask_final.flatten()
    print("正在进行畸变校正...")
    pts1_valid = pts1[flat_mask].astype(np.float32).copy()
    pts2_valid = pts2[flat_mask].astype(np.float32).copy()

    print(f"剔除边缘无效点后，剩余计算点数: {len(pts1_valid)}")

    # 接下来只对 pts1_valid 和 pts2_valid 进行 undistortPoints 和 triangulatePoints
    pts1_ud = cv2.undistortPoints(pts1_valid, K1, dist1, P=K1)
    pts2_ud = cv2.undistortPoints(pts2_valid, K2, dist2, P=K2)

    # --- 6. 执行三角化 (Triangulation) ---
    print("正在进行三维空间三角化...")
    # cv2.triangulatePoints 需要 2xN 的输入
    points_4d = cv2.triangulatePoints(P1, P2, pts1_ud.reshape(-1, 2).T, pts2_ud.reshape(-1, 2).T)

    # 归一化齐次坐标得到 (X, Y, Z)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d.T  # 变为 Nx3

    # ----------------- 附加步骤：方法1 - 统计滤波去尖刺 -----------------
    print("正在进行统计离群点剔除 (SOR)...")

    # 1. 将 numpy 数组转换为 Open3D 的 PointCloud 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # 2. 执行统计滤波
    # nb_neighbors: 考察周围的多少个点来计算平均距离 (建议 20-50)
    # std_ratio: 阈值。越小删的点越多，越大删的点越少 (建议 2.0 - 3.0)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)

    # 3. 提取过滤后的“干净”点云
    pcd_clean = pcd.select_by_index(ind)

    points_3d = np.asarray(pcd_clean.points)

    # --- 7. 结果过滤与保存 ---
    # 过滤无效点（比如离相机太远或太近的点，根据你的实验台调整）
    z_vals = points_3d[:, 2]
    mask = (z_vals > 50) & (z_vals < 500)
    valid_points = points_3d[mask]

    output_path = os.path.join(data_dir, "final_scientific_cloud.xyz")
    np.savetxt(output_path, valid_points, fmt='%.4f %.4f %.4f')

    print("-" * 30)
    print(f"重建完成！")
    print(f"原始点数: {len(points_3d)} | 有效点数: {len(valid_points)}")
    print(f"平均深度 Z: {np.mean(valid_points[:, 2]):.2f} mm")
    print(f"存储路径: {output_path}")
    print("-" * 30)


if __name__ == "__main__":
    reconstruct_3d_pro()