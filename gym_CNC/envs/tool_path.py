import numpy as np
import pygame
from numpy.linalg import norm

from gym_CNC.envs.road_sides import edge_points, unit_normals

EPS = 1e-6


""" 这个是用于计算CNC强化学习环境中action、state、reward等各个量的类 """


class ToolPath:
    # __init__函数：生成一个由刀位点组成的刀路，定义边界宽度、点的个数、屏幕宽度、屏幕高度等量
    def __init__(self, points=None, band_width=0.02):
        self.band_width = band_width
        if points is None:
            self._generate_points()
        else:
            self.pyg_points = points
        self.N = self.points.shape[0]
        self.screen_width = 800
        self.screen_height = 800
        self.theta = 0  # 这一回合刀运动的方向
        self.len = 0  # 这一回合刀运动的距离
        self.j = 1  # 代表当前点所在的是第几条线段，从1开始计数
        self.state = None
        self.reward = 0
        self.episode_step = 0  # 记录每个episode刀走的步数
        self.terminated = False
        self.truncated = False

    # _generate_points函数：加载并解析G代码文件，调用函数生成理想轨迹点和边界轨迹点，并将其转化为屏幕上的坐标点
    def _generate_points(self):
        # 加载并解析G代码文件
        data = np.loadtxt("data/P_norm.txt")
        self.points = data
        left_edge, right_edge = edge_points(self.points, self.band_width)
        # 添加起点和终点的边缘点
        start_normal = unit_normals(self.points[:2])[0]
        end_normal = unit_normals(self.points[-2:])[0]
        self.points_l = np.vstack(
            (
                self.points[0] + start_normal * self.band_width,
                left_edge,
                self.points[-1] + end_normal * self.band_width,
            )
        )
        self.points_r = np.vstack(
            (
                self.points[0] - start_normal * self.band_width,
                right_edge,
                self.points[-1] - end_normal * self.band_width,
            )
        )
        self.pyg_points = convert_points(self.points)
        self.pyg_points_l = convert_points(self.points_l)
        self.pyg_points_r = convert_points(self.points_r)
        self.position = self.points[0]
        self.last_position = self.points[0]  # 上个时刻的位置点坐标
        self.second_last_position = self.points[0]  # 上上个时刻的位置点坐标
        self.P = [self.position]

    def find_nearest_line_segment(self):
        # 这里的j_curr代表的是第几条线段，最小是1（第1条线段），它是由第0个点和第1个点组成
        d_min = point_to_line_segment_distance(
            self.position,
            self.points[self.j - 1],
            self.points[self.j],
        )
        self.j += 1
        while self.j < self.N:
            d = point_to_line_segment_distance(
                self.position,
                self.points[self.j - 1],
                self.points[self.j],
            )
            if d <= d_min:
                d_min = d
                self.j += 1
            else:
                break
        self.j -= 1
        self.reward = -d_min

    def calculate_next_position(self, action):
        """
        计算下一个时刻的位置坐标。

        参数:
        - position: 当前时刻的坐标 (x_t, y_t) 作为 1x2 的 NumPy 数组
        - theta_t: 当前时刻的方向角度 theta_t
        - len_t: 上一时刻走过的长度
        - action: 包含角度变化量 theta_t' 和长度变化量 l_t' 的动作，作为 1x2 的 NumPy 数组

        返回:
        - next_position: 下一个时刻的位置坐标 (x_{t+1}, y_{t+1}) 作为 1x2 的 NumPy 数组
        """
        # 解包动作中的角度变化和长度变化
        theta_change, length_change = action

        # 计算下一时刻的总角度和总长度
        theta_next = self.theta + theta_change
        l_next = self.len + length_change

        # 根据当前位置、角度变化和长度变化计算下一位置
        x_next = self.position[0] + l_next * np.cos(theta_next)
        y_next = self.position[1] + l_next * np.sin(theta_next)

        self.second_last_position = self.last_position
        self.last_position = self.position
        self.position = np.array([x_next, y_next])
        self.P.append(self.position)

    def calculate_angle_deviation(self):
        """
        计算角偏差τ。

        参数:
        - j: 当前点所在的线段索引
        - position: 当前点的坐标，作为 1x2 的 NumPy 数组
        - last_position: 上一时刻点的坐标，作为 1x2 的 NumPy 数组

        返回:
        - tau: 角偏差τ

        如果τ是正值，意味着需要逆时针旋转（从当前方向到理想方向）来对齐到理想路径方向；
        负值则相反。
        """
        # 计算理想路径方向的角度 theta_理想
        vector_ideal = self.points[self.j] - self.points[self.j - 1]
        theta_ideal = np.arctan2(vector_ideal[1], vector_ideal[0])

        # 计算上一时刻方向的角度 theta_上一时刻
        vector_last = self.position - self.last_position
        theta_last = np.arctan2(vector_last[1], vector_last[0])

        # 计算角偏差 tau
        tau = theta_ideal - theta_last

        # 确保 tau 在 -pi 到 pi 的范围内
        tau = (tau + np.pi) % (2 * np.pi) - np.pi

        return np.array(tau)

    def find_closest_intersection_with_limit(
        self,
        ray_origin,
        ray_direction,
        nearest_segment_index,
        epsilon,
        num_segments_before=3,
        num_segments_after=4,
    ):
        closest_point = None
        min_distance = float("inf")

        # Calculate the start and end indices of the segments we want to check
        start_index = max(1, nearest_segment_index - num_segments_before)
        end_index = min(self.N - 1, nearest_segment_index + num_segments_after + 1)

        # Normalize the direction vector
        ray_direction = np.array(ray_direction) / np.linalg.norm(ray_direction)

        # Iterate over the specified range of segments in the boundary
        for i in range(start_index, end_index):
            p1 = np.array(self.points[i - 1])
            p2 = np.array(self.points[i])
            intersection = line_intersect(
                ray_origin, ray_origin + ray_direction * 10000, p1, p2
            )

            if intersection is not None:
                vector_to_intersection = intersection - ray_origin
                if (
                    np.dot(vector_to_intersection, ray_direction) > 0
                ):  # Check if the point is in the direction of the ray
                    distance = np.linalg.norm(vector_to_intersection)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = intersection

        # If no intersection was found or if the closest distance is greater than epsilon, return epsilon
        if closest_point is None or min_distance > epsilon:
            return epsilon
        else:
            return min_distance

    # 该函数输入当前时刻主方向的角度，返回雷达中所含的9个方向的单位向量(输入的theta为numpy格式)
    def calculate_rader_vectors(self):
        vectors = []
        vectors.append(np.array([np.cos(self.theta), np.sin(self.theta)]))
        angle_increment = np.pi / (2 * 4)
        angle1 = self.theta
        angle2 = self.theta
        for i in range(4):
            angle1 = angle1 - angle_increment
            angle2 = angle2 - angle_increment
            vectors.append(np.array([np.cos(angle1), np.sin(angle1)]))
            vectors.append(np.array([np.cos(angle2), np.sin(angle2)]))

        return vectors

    def render(self, cnc_env):
        screen = cnc_env.screen
        background_color = (255, 255, 255)
        screen.fill(background_color)  # 使用白色清屏
        # 设置颜色
        black = (0, 0, 0)
        blue = (0, 0, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        tool_path = convert_points(self.P)
        pygame.draw.lines(screen, black, False, self.pyg_points, 1)
        pygame.draw.lines(screen, red, False, self.pyg_points_l, 1)
        pygame.draw.lines(screen, red, False, self.pyg_points_r, 1)
        pygame.draw.lines(screen, green, False, tool_path, 1)

        for point in tool_path:
            pygame.draw.circle(screen, green, point, 1)

    def init_state(self):
        state1 = np.array([0, 0, 0])
        vector_ideal = self.points[1] - self.points[0]
        self.theta = np.arctan2(vector_ideal[1], vector_ideal[0])

        state_vector = []
        vectors = self.calculate_rader_vectors()  # 先求出这一时刻的9个雷达方向的单位向量
        for i in range(len(vectors)):
            distance = self.find_closest_intersection_with_limit(
                self.position,
                vectors[i],
                self.j,
                0.45,
            )
            state_vector.append(distance)
        self.state = np.concatenate((state1, state_vector))

        return self.state

    def action(self, action):
        self.calculate_next_position(action)  # 求出采取该动作之后的该点位置以及当前刀运动的方向角和走的长度
        self.find_nearest_line_segment()  # 更新当前点所处的线段，以及最小距离

        # 下面求状态state，分为三个部分
        self.state = action
        # 计算state第二部分：当前运动方向与理想方向的偏移
        tau = self.calculate_angle_deviation()
        state_vector = [tau]
        # 计算state第三部分：9个方向分别与边界点的距离
        vectors = self.calculate_rader_vectors()  # 先求出这一时刻的9个雷达方向的单位向量
        for i in range(len(vectors)):
            distance = self.find_closest_intersection_with_limit(
                self.position,
                vectors[i],
                self.j,
                0.03,
            )
            state_vector.append(distance)
        self.state = np.concatenate((action, state_vector))

        self.episode_step += 1  # 每采取一个行动，就+1表示走过了一个时间步
        if self.episode_step == 2000:
            self.truncated = True

        return self.state, self.reward

    def is_done(self):
        d = self.position - self.points[self.N - 1]
        distance = np.linalg.norm(d)
        if self.j == self.N - 1 and distance < self.band_width:
            self.terminated = True

        return self.terminated, self.truncated

    # def check_radar_for_draw(self, radians_offset):
    #     rader_length = 0
    #     # 直接使用弧度制的角度计算，考虑到pygame中y轴的反向
    #     x = int(self.center[0] + math.cos(self.theta + radians_offset) * rader_length)
    #     # 使用负的sin值来适应pygame的坐标系
    #     y = int(self.center[1] - math.sin(self.theta + radians_offset) * rader_length)
    #
    #     while (
    #         not self.map.get_at((x, y)) == (255, 255, 255, 255) and rader_length < 2000
    #     ):
    #         rader_length += 1
    #         x = int(
    #             self.center[0] + math.cos(self.theta + radians_offset) * rader_length
    #         )
    #         y = int(
    #             self.center[1] - math.sin(self.theta + radians_offset) * rader_length
    #         )
    #
    #     dist = int(math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
    #     self.radars_for_draw.append([(x, y), dist])


def calculate_distance_and_angle(p, q, v):
    pq = q - p
    distance = np.linalg.norm(pq)
    cos_angle = np.dot(pq, v) / (distance * np.linalg.norm(v) + EPS)
    angle = np.arccos(cos_angle)

    # 使用叉积的符号直接确定角度的正负
    if np.cross(pq, v) < 0:
        angle = -angle

    return distance, angle


def point_to_line_segment_distance(p, a, b):
    """
    Calculates the distance between a point and a line segment defined by two points.

    Args:
        p (numpy.ndarray): The point to calculate the distance from.
        a (numpy.ndarray): The first point defining the line segment.
        b (numpy.ndarray): The second point defining the line segment.

    Returns:
        float: The distance between the point and the line segment.
    """
    proj = np.dot(p - a, b - a) / norm(b - a) ** 2
    if proj < 0:
        d = norm(p - a)
    elif proj > 1:
        d = norm(p - b)
    else:
        d = norm(np.cross(p - a, p - b)) / norm(b - a)
    return d


def convert_points(points, screen_width=800, screen_height=800):
    """
    Converts points in the range [-1, 1] x [-1, 1] to points on a pygame screen.

    Args:
        points (numpy.ndarray): The points to convert.
        screen_width (int): The width of the pygame screen.
        screen_height (int): The height of the pygame screen.

    Returns:
        numpy.ndarray: The converted points.
    """
    pts = np.array(points)
    x = ((pts[:, 0] + 1) * screen_width / 2).astype(int)
    y = ((1 - pts[:, 1]) * screen_height / 2).astype(int)
    return np.column_stack((x, y))


def convert_back(points, screen_width=800, screen_height=800):
    """
    Converts points on a pygame screen to points in the range [-1, 1] x [-1, 1].

    Args:
        points (numpy.ndarray): The points to convert.
        screen_width (int): The width of the pygame screen.
        screen_height (int): The height of the pygame screen.

    Returns:
        numpy.ndarray: The converted points.
    """
    x = (2 * points[:, 0] / screen_width) - 1
    y = 1 - (2 * points[:, 1] / screen_height)
    return np.column_stack((x, y))


# Corrected line_intersect function to check for the segment intersection properly
def line_intersect(p1, p2, p3, p4):
    # Line p1p2 represented as a1x + b1y = c1
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    # Line p3p4 represented as a2x + b2y = c2
    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2 * p3[0] + b2 * p3[1]

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        # The lines are parallel, no intersection
        return None

    # Solve the system for x and y
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    intersect_point = np.array([x, y])

    # Check if the intersection point is on both segments
    if (
        min(p1[0], p2[0]) <= x <= max(p1[0], p2[0])
        and min(p1[1], p2[1]) <= y <= max(p1[1], p2[1])
        and min(p3[0], p4[0]) <= x <= max(p3[0], p4[0])
        and min(p3[1], p4[1]) <= y <= max(p3[1], p4[1])
    ):
        return intersect_point

    return None
