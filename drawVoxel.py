import numpy as np
import open3d as o3d

def drawVoxel(data):
    data = np.transpose(data, (1, 2, 0)) # 그림 그릴땐 맨 뒤가 높이
    print("데이터 로드 및 변환 완료")

    occupied_indices = np.argwhere(data != 0)
    if len(occupied_indices) == 0:
        print("경고 : 데이터가 모두 0입니다. (빈맵). 그릴 것이 없습니다.")
        return
    
    values = data[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]]
    colors = np.zeros((occupied_indices.shape[0], 3))

    colors[values==1] = [0.3, 0.3, 0.3]
    colors[values==2] = [0.0, 0.0, 1.0]
    colors[values==3] = [1.0, 0.0, 0.0]

    # for i, (x, y, z) in enumerate(occupied_indices):
    #     val = data[x, y, z]
    #     if val == 1: colors[i] = [0.3, 0.3, 0.3] # Road - 회색
    #     elif val == 2: colors[i] = [0, 0, 1.0] # Car - 파란색
    #     elif val == 3: colors[i] = [1.0, 0, 0] # Obstacle - 빨간색
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(occupied_indices)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)

    print("Open3D 창 열기")

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([voxel_grid, mesh_frame], window_name="My Voxel Map", width=800, height=600)