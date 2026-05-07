import numpy as np
import pyransac3d as pyrsc
import os

def process_cylinder_cloud():
    # --- 参数设置 ---
    input_cloud_path = "/home/wenhao/bishe_code/2DTrans3D_result/final_scientific_cloud.xyz" # 替换为你的点云路径
    save_dir = "/home/wenhao/bishe_code/2DTrans3D_result/final_scientific_cloud"
    os.makedirs(save_dir, exist_ok=True)

    # 距离阈值 (关键参数)：点距离拟合圆柱面超过 1.5mm 的视为噪点，你可以根据尖刺的粗细调整这个值
    dist_threshold = 1.5

    print("正在加载点云...")
    points = np.loadtxt(input_cloud_path)
    print(f"加载点云成功。点数: {len(points)}")

    # --- 核心算法：RANSAC 圆柱拟合与去噪 ---
    print("正在执行 3D RANSAC 圆柱拟合与去噪...")

    # 初始化圆柱拟合器
    cylinder = pyrsc.Cylinder()

    # 执行拟合
    # 输入 points。max_dist 是判定内点的距离阈值
    center, axis, radius, inliers = cylinder.fit(points, thresh=dist_threshold, maxIteration=2000)

    # 计算有效（内点）数据和噪点数据
    valid_points = points[inliers]

    # --- 结果分析与保存 ---
    print("-" * 30)
    print(f"重建圆柱半径 (RANSAC) 为: {radius:.4f} mm")
    print(f"圆柱中心位置: {center}")
    print(f"圆柱轴线方向: {axis}")
    print(f"RANSAC 提纯效果: {len(points)} 个原始点，剔除 {len(points) - len(valid_points)} 个噪点。")
    print("-" * 30)

    # 保存去噪后的点云
    save_path = os.path.join(save_dir, 'cleaned_cylinder_cloud.xyz')
    np.savetxt(save_path, valid_points, fmt='%.4f %.4f %.4f')
    print(f"去噪点云已保存至: {save_path}")

if __name__ == "__main__":
    process_cylinder_cloud()