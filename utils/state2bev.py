import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def vehicle_coordinate_sys(base_position, base_speed, base_heading, position, speed=None, heading=None):
    def direction(heading) -> np.ndarray:
        return np.array([np.cos(heading), np.sin(heading)])

    delta_x = position[0] - base_position[0]
    delta_y = position[1] - base_position[1]

    # 转换坐标系：旋转-A的朝向角
    rot_matrix = np.array([
        [np.cos(-base_heading), -np.sin(-base_heading)],
        [np.sin(-base_heading), np.cos(-base_heading)]
    ])

    ego_vel = direction(base_heading) * base_speed
    rel_position = rot_matrix @ np.array([delta_x, delta_y])

    rel_velocity = None
    rel_yaw = None

    if speed is not None:
        bv_vel = direction(heading) * speed
        rel_velocity = rot_matrix @ np.array([bv_vel[0] - ego_vel[0], bv_vel[1] - ego_vel[1]])

    if heading is not None:
        # 计算车辆B相对于A的朝向
        rel_yaw = heading - base_heading
        # 调整朝向角到范围 [-pi, pi]
        rel_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi

    return rel_position, rel_velocity, rel_yaw


def absolute_coordinate_sys(
        base_position: np.ndarray,
        base_speed: float,
        base_heading: float,
        rel_position: np.ndarray,
        rel_velocity: np.ndarray = None,
        rel_yaw: float = None
):

    rot_matrix = np.array([
        [np.cos(base_heading), -np.sin(base_heading)],
        [np.sin(base_heading), np.cos(base_heading)]
    ])
    delta_global = rot_matrix @ rel_position
    abs_position = base_position + delta_global

    # 2. 计算绝对速度
    abs_velocity = None
    if rel_velocity is not None:
        # 基础车辆的速度向量
        base_vel = np.array([
            np.cos(base_heading) * base_speed,
            np.sin(base_heading) * base_speed
        ])
        # 相对速度转换为绝对速度
        abs_rel_vel = rot_matrix @ rel_velocity
        abs_velocity = abs_rel_vel + base_vel

    # 3. 计算绝对航向角
    abs_yaw = None
    if rel_yaw is not None:
        abs_yaw = base_heading + rel_yaw
        # 调整到 [-π, π]
        abs_yaw = (abs_yaw + np.pi) % (2 * np.pi) - np.pi

    return abs_position, abs_velocity, abs_yaw


def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


def polygon_xy_from_state(x, y, psi_rad, width, length):
    lowleft = (x - length / 2., y - width / 2.)
    lowright = (x + length / 2., y - width / 2.)
    upright = (x + length / 2., y + width / 2.)
    upleft = (x - length / 2., y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([x, y]), yaw=psi_rad)


def draw_vehicle(ax, cx, cy, heading, vehicle_length, vehicle_width, color='blue'):
    rect = matplotlib.patches.Polygon(polygon_xy_from_state(cx, cy, heading, vehicle_width, vehicle_length), closed=True,
                                      zorder=20, color=color)
    ax.add_patch(rect)

    # # 绘制车辆：矩形表示
    # ax.add_patch(plt.Rectangle((polygon[0][0], polygon[0][1]), vehicle_length, vehicle_width,
    #                            angle=heading * 180 / np.pi, color=color))  # 绿色矩形表示车辆
    #
    # # 绘制朝向箭头
    # ax.arrow(cx, cy , vehicle_length * np.cos(heading) * 0.5, vehicle_length * np.sin(heading) * 0.5,
    #          head_width=1, head_length=1, fc='r', ec='r')  # 红色箭头表示朝向


def draw_pedestrian(ax, x, y, heading):
    # 绘制车辆：矩形表示
    vehicle_width = 0.5
    vehicle_length = 0.5
    ax.scatter(x, y, c='b', s=100, label='Pedestrian')  # 蓝色圆点表示行人

    # 绘制朝向箭头
    ax.arrow(x - vehicle_length / 2, y - vehicle_width / 2, vehicle_length * np.cos(heading), vehicle_length * np.sin(heading),
             head_width=5, head_length=5, fc='r', ec='r')  # 红色箭头表示朝向