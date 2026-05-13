import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def convert_csv_to_colored_ply(input_csv, output_ply):
    # 1. 读取数据
    df = pd.read_csv(input_csv)

    # 2. 提取 XYZ 坐标
    try:
        points = df[['X/mm', 'Y/mm', 'Z/mm']].values
    except KeyError:
        print("错误：CSV中未找到 X/mm, Y/mm, Z/mm 列")
        return

    # 3. 计算 Z 轴的深度颜色映射
    z_values = points[:, 2]
    z_min = z_values.min()
    z_max = z_values.max()

    # 归一化 Z 值到 [0, 1] 范围
    # 这样最小的深度是 0，最大的深度是 1
    z_norm = (z_values - z_min) / (z_max - z_min + 1e-6)

    # 4. 使用 matplotlib 的 Colormap (例如 'jet' 或 'viridis')
    # 'jet' 是经典的彩虹色（蓝->绿->黄->红），'viridis' 则是现代且感知均匀的颜色
    cmap = plt.get_cmap('jet')
    colors = cmap(z_norm)[:, :3]  # 只取 RGB，去掉 Alpha 通道

    # 5. 创建并配置 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 6. 导出包含颜色的 PLY 文件
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"彩色点云已导出至: {output_ply}")

    # 7. 可视化
    o3d.visualization.draw_geometries([pcd], window_name="Depth Colored Point Cloud")


if __name__ == "__main__":
    convert_csv_to_colored_ply('/home/wenhao/bishe_code/DIC/Camera00_00000001_00.csv', '/home/wenhao/bishe_code/DIC/1')