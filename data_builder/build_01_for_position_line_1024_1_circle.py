if __name__ == "__main__":
    import os
    import time
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from SimPEG import maps
    from SimPEG.potential_fields import magnetics
    from discretize import TensorMesh
    from global_args import proj_path
    from utils.receiver_line_utils import *
    from utils.mesh_model_builder import get_indices_ellipsoid

# 场景设定：定位距离坐标中心点2500m范围内的水下磁性目标物
# 在距离海面100m的高度，飞机沿以坐标中心为中心、半径300m的圆周上采集数据，采样点间距1m，共采集1024个点。
# 生成格式：
# data: 1024个磁异常数值
# label: 7个数值，对应潜艇的x，y，z坐标，磁化率，长宽高。
# 本文件使用到了SimPEG磁正演，建议不看或者最后看

# File params
dataSize = 6500  # 生出数据条数
interrupt = 0  # 生成间断后想继续生成可以调节该参数，使开始索引为 interrupt
# 输出路径
label_path = os.path.join(proj_path, 'datasets/train_1024_1_circle_pos/train/label/')
data_path = os.path.join(proj_path, 'datasets/train_1024_1_circle_pos/train/data/')
is_plot = False

# Mesh params 形状参数
dh = 1
# 目标物体的形状可变范围，需根据具体情况删减并调节。例如圆形只需要一个 radius_range 而非以下三种
length_range, width_range, height_range = (95, 115), (9, 12), (9, 12)
shape_range_collection = (length_range, width_range, height_range)
# 目标物体的最大可变形状的三轴范围，用于生成最贴近目标体的 mesh
max_length, max_width, max_height = length_range[1], width_range[1], height_range[1]
hx_v, hy_v, hz_v = math.ceil(max_length / dh), math.ceil(max_width / dh), math.ceil(max_height / dh)
hx, hy, hz = [(dh, hx_v)], [(dh, hy_v)], [(dh, hz_v)]
mesh = TensorMesh([hx, hy, hz], "CCN")

# Distance params 距离参数
# 以探测中心为原点 z为深度
# 目标物体距离探测中心的可变距离范围
x_range, y_range, z_range = (-2500, 2500), (-2500, 2500), (0, 400)

# 地磁场 背景磁化率 目标磁化率
background_susceptibility = 0.0
# 南海某点背景地场
# Latitude: 18° 18' 40.3" N
# Longitude: 118° 12' 45.3" E
# # (strength, inclination, declination)
inducing_field = (43000, 25.48333, -3.16666)
sensitivity_range = (80, 80)

# 探测飞行高度
# 高度调研 100m
receiver_height = 100
# 飞行探测接受点函数
receiver_func = get_circle_line_by_num
d_length = 1  # 测点间距
re_point = 1024  # 测点数

# 参数整合
shape = [np.random.rand(1, dataSize) * (i[1] - i[0]) + i[0]
         for i in shape_range_collection]
x1 = np.random.rand(1, dataSize) * (x_range[1] - x_range[0]) + x_range[0]
y1 = np.random.rand(1, dataSize) * (y_range[1] - y_range[0]) + y_range[0]
z1 = np.random.rand(1, dataSize) * (z_range[1] - z_range[0]) + z_range[0]
sensitivity1 = np.random.rand(1, dataSize) * (sensitivity_range[1] - sensitivity_range[0]) + sensitivity_range[0]
randoms = np.vstack((x1, y1, z1, sensitivity1, *shape))
x_idx, y_idx, z_idx, sensitivity_idx = 0, 1, 2, 3  # 几个重要参数在 randoms 中的索引，必须设置
length_idx, width_idx, height_idx = 4, 5, 6


def get_model(input_random):
    # 预设模型为空心椭圆
    # model
    result_model = background_susceptibility * np.ones(mesh.n_cells)
    # 0, 0, (-hz_v * dh / 2.0) 将目标物设定在 mesh 的三轴中心上
    ind_block = get_indices_ellipsoid((0, 0, (-hz_v * dh / 2.0)),
                                      input_random[length_idx] / 2.0,
                                      input_random[width_idx] / 2.0,
                                      input_random[height_idx] / 2.0,
                                      mesh.cell_centers)
    ind_block2 = get_indices_ellipsoid((0, 0, -hz_v * dh / 2.0),
                                       input_random[length_idx] / 2.0 * 0.7,
                                       input_random[width_idx] / 2.0 * 0.7,
                                       input_random[height_idx] / 2.0 * 0.7,
                                       mesh.cell_centers)
    result_model[ind_block] = random[sensitivity_idx]
    result_model[ind_block2] = 0
    return result_model, ind_block2 ^ ind_block


# Auto make directory
if not os.path.isdir(label_path): os.makedirs(label_path)
if not os.path.isdir(data_path): os.makedirs(data_path)
for idx, random in enumerate(np.transpose(randoms)):
    time_start = time.time()

    # Interrupt index
    idx = idx + interrupt

    model, ind_block = get_model(random)

    # Suppose all cells are below sea level
    # Find the indices for the active mesh cells (e.g. cells below surface)
    ind_active = np.full(mesh.n_cells, True, dtype='bool')
    nC = int(ind_active.sum())
    x, y, z = 0 - random[x_idx], 0 - random[y_idx], receiver_height + random[z_idx]
    receiver_locations = np.c_[receiver_func(n=re_point, x=x, y=y, z=z, d_length=d_length)]
    receiver_list = [magnetics.receivers.Point(receiver_locations, components=["tmi"])]
    source_field = magnetics.sources.SourceField(receiver_list=receiver_list, parameters=inducing_field)
    survey = magnetics.survey.Survey(source_field)
    simulation = magnetics.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        chiMap=maps.IdentityMap(nP=nC),
        ind_active=ind_active,
        store_sensitivities="forward_only",
    )

    # Plot Model
    if is_plot:
        fig = plt.figure(figsize=(9, 4))

        plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
        ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
        mesh.plot_slice(
            plotting_map * model,
            normal="Y",
            ax=ax1,
            ind=int(mesh.shape_cells[1] / 2),
            grid=True,
            clim=(np.min(model), np.max(model)),
        )
        # mesh.plot_3d_slicer(plotting_map * model)
        ax1.set_title("Model slice at y = 0 m")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("z (m)")

        ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
        norm = mpl.colors.Normalize(vmin=np.min(model), vmax=np.max(model))
        cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical")
        cbar.set_label("Magnetic Susceptibility (SI)", rotation=270, labelpad=15, size=12)

        plt.show()

    dpred = simulation.dpred(model)

    result = []
    # 加 (hz_v * dh / 2.0) 是为了将输出的物体中心从 mesh 正上方下降到 mesh 正中心
    for r_idx, i in enumerate(random):
        if r_idx == z_idx:
            result.append(i + (hz_v * dh / 2.0))
        else:
            result.append(i)

    # File params
    label_file_name = os.path.join(label_path, str(idx) + ".txt")
    data_file_name = os.path.join(data_path, str(idx) + ".txt")
    if os.path.exists(label_file_name) or os.path.exists(data_file_name):
        raise FileExistsError(label_file_name + " or " + data_file_name + " have existed!")
    np.savetxt(label_file_name, result, fmt="%.4e")
    np.savetxt(data_file_name, dpred, fmt="%.4e")

    print('Count ' + str(idx) + ' Time cost = %fs' % (time.time() - time_start))
