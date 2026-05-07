import open3d as o3d
import numpy as np

# 1. 加载你刚刚生成的点云文件
# 假设文件路径是 /home/wenhao/bishe_code/cloud_output_perfect.xyz
pcd_path = "./2DTrans3D_result/final_scientific_cloud.xyz"
points = np.loadtxt(pcd_path)

# 2. 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 3. 可选：估计法线（让点云看起来更有立体感）
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))

# 4. 可选：给点云上色 (比如根据深度 Z 值上色)
# 获取 Z 轴范围
z_values = points[:, 2]
z_min, z_max = np.min(z_values), np.max(z_values)
# 简单的归一化颜色映射（从蓝到红）
colors = np.zeros((points.shape[0], 3))
colors[:, 0] = (z_values - z_min) / (z_max - z_min)  # R
colors[:, 2] = 1 - colors[:, 0]                     # B
pcd.colors = o3d.utility.Vector3dVector(colors)

# 5. 启动可视化窗口
print("正在打开可视化窗口... 您可以用鼠标旋转和缩放。")
o3d.visualization.draw_geometries([pcd],
                                  window_name="毕业设计-3D点云可视化",
                                  width=1200, height=800,
                                  left=50, top=50,
                                  mesh_show_back_face=True)