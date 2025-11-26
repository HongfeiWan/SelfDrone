import numpy as np
import open3d as o3d

# ------------------------------
# 1. 读取点云并按高度着色
# ------------------------------
pcd = o3d.io.read_point_cloud("bunny.ply")

points = np.asarray(pcd.points)
z_vals = points[:, 2]
colors = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
pcd.colors = o3d.utility.Vector3dVector(
    np.stack([colors, colors, colors], axis=1)
)

# ------------------------------
# 2. 点云 → 体素
# ------------------------------
voxel_size = 0.005
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

# ------------------------------
# 3. VoxelGrid → 点云中心（还原点云）
# ------------------------------

voxels = voxel_grid.get_voxels()
restored_points = []
restored_colors = []

for v in voxels:
    idx = np.array(v.grid_index, dtype=float)
    center = voxel_grid.origin + (idx + 0.5) * voxel_size
    restored_points.append(center)

    if hasattr(v, 'color'):
        restored_colors.append(v.color)
    else:
        restored_colors.append([1.0, 0.0, 0.0])  # 默认红色（便于区分）

restored_points = np.array(restored_points)
restored_colors = np.array(restored_colors)

# ------------------------------
# 4. 添加高斯噪声
# ------------------------------
sigma = voxel_size * 0.25
noise = np.random.normal(scale=sigma, size=restored_points.shape)
restored_points_noisy = restored_points + noise

# ------------------------------
# 5. 构建还原后的点云（红色）
# ------------------------------
pcd_restored = o3d.geometry.PointCloud()
pcd_restored.points = o3d.utility.Vector3dVector(restored_points_noisy)

# 用统一的颜色（红色）
pcd_restored.paint_uniform_color([1.0, 0.0, 0.0])

# ------------------------------
# 6. 原始点云设为蓝色（便于区分）
# ------------------------------
pcd_original_vis = o3d.geometry.PointCloud(pcd)   # 复制
pcd_original_vis.paint_uniform_color([0.0, 0.4, 1.0])

# ------------------------------
# 7. 同时绘制两个点云
# ------------------------------
o3d.visualization.draw_geometries([pcd_original_vis, pcd_restored])
