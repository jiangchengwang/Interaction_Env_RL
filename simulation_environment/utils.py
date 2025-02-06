from __future__ import annotations

import copy
import importlib
import itertools
import os
from typing import Callable, List, Sequence, Tuple, Union
from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np
from utils.cubic_spline import Spline2D

# # Useful types
# Vector = Union[np.ndarray, Sequence[float]]
# Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
# Interval = Union[
#     np.ndarray,
#     Tuple[Vector, Vector],
#     Tuple[Matrix, Matrix],
#     Tuple[float, float],
#     List[Vector],
#     List[Matrix],
#     List[float],
# ]


def do_every(duration: float, timer: float) -> bool:
    return duration < timer


def lmap(v: float, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

def remap(v, x, y):
    return y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])

def get_class_path(cls: Callable) -> str:
    return cls.__module__ + "." + cls.__qualname__


def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def constrain(x: float, a: float, b: float) -> np.ndarray:
    return np.clip(x, a, b)


def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x >= 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def point_in_rectangle(point, rect_min, rect_max) -> bool:
    """
    Check if a point is inside a rectangle

    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return (
        rect_min[0] <= point[0] <= rect_max[0]
        and rect_min[1] <= point[1] <= rect_max[1]
    )


def point_in_rotated_rectangle(
    point: np.ndarray, center: np.ndarray, length: float, width: float, angle: float
) -> bool:
    """
    Check if a point is inside a rotated rectangle

    :param point: a point
    :param center: rectangle center
    :param length: rectangle length
    :param width: rectangle width
    :param angle: rectangle angle [rad]
    :return: is the point inside the rectangle
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, (-length / 2, -width / 2), (length / 2, width / 2))


def point_in_ellipse(
    point, center, angle: float, length: float, width: float
) -> bool:
    """
    Check if a point is inside an ellipse

    :param point: a point
    :param center: ellipse center
    :param angle: ellipse main axis angle
    :param length: ellipse big axis
    :param width: ellipse small axis
    :return: is the point inside the ellipse
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(
    rect1, rect2
) -> bool:
    """
    Do two rotated rectangles intersect?

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    :return: do they?
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def rect_corners(
    center: np.ndarray,
    length: float,
    width: float,
    angle: float,
    include_midpoints: bool = False,
    include_center: bool = False,
) -> list[np.ndarray]:
    """
    Returns the positions of the corners of a rectangle.
    :param center: the rectangle center
    :param length: the rectangle length
    :param width: the rectangle width
    :param angle: the rectangle angle
    :param include_midpoints: include middle of edges
    :param include_center: include the center of the rect
    :return: a list of positions
    """
    center = np.array(center)
    half_l = np.array([length / 2, 0])
    half_w = np.array([0, width / 2])
    corners = [-half_l - half_w, -half_l + half_w, +half_l + half_w, +half_l - half_w]
    if include_center:
        corners += [[0, 0]]
    if include_midpoints:
        corners += [-half_l, half_l, -half_w, half_w]

    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return (rotation @ np.array(corners).T).T + np.tile(center, (len(corners), 1))


def has_corner_inside(
    rect1, rect2
) -> bool:
    """
    Check if rect1 has a corner inside rect2

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    return any(
        [
            point_in_rotated_rectangle(p1, *rect2)
            for p1 in rect_corners(*rect1, include_midpoints=True, include_center=True)
        ]
    )


def project_polygon(polygon, axis) -> tuple[float, float]:
    min_p, max_p = None, None
    for p in polygon:
        projected = p.dot(axis)
        if min_p is None or projected < min_p:
            min_p = projected
        if max_p is None or projected > max_p:
            max_p = projected
    return min_p, max_p


def interval_distance(min_a: float, max_a: float, min_b: float, max_b: float):
    """
    Calculate the distance between [minA, maxA] and [minB, maxB]
    The distance will be negative if the intervals overlap
    """
    return min_b - max_a if min_a < min_b else min_a - max_b


def are_polygons_intersecting(
    a, b, displacement_a, displacement_b
) -> tuple[bool, bool, np.ndarray | None]:
    """
    Checks if the two polygons are intersecting.

    See https://www.codeproject.com/Articles/15573/2D-Polygon-Collision-Detection

    :param a: polygon A, as a list of [x, y] points
    :param b: polygon B, as a list of [x, y] points
    :param displacement_a: velocity of the polygon A
    :param displacement_b: velocity of the polygon B
    :return: are intersecting, will intersect, translation vector
    """
    intersecting = will_intersect = True
    min_distance = np.inf
    translation, translation_axis = None, None
    for polygon in [a, b]:
        for p1, p2 in zip(polygon, polygon[1:]):
            normal = np.array([-p2[1] + p1[1], p2[0] - p1[0]])
            normal /= np.linalg.norm(normal)
            min_a, max_a = project_polygon(a, normal)
            min_b, max_b = project_polygon(b, normal)

            if interval_distance(min_a, max_a, min_b, max_b) > 0:
                intersecting = False

            velocity_projection = normal.dot(displacement_a - displacement_b)
            if velocity_projection < 0:
                min_a += velocity_projection
            else:
                max_a += velocity_projection

            distance = interval_distance(min_a, max_a, min_b, max_b)
            if distance > 0:
                will_intersect = False
            if not intersecting and not will_intersect:
                break
            if abs(distance) < min_distance:
                min_distance = abs(distance)
                d = a[:-1].mean(axis=0) - b[:-1].mean(axis=0)  # center difference
                translation_axis = normal if d.dot(normal) > 0 else -normal

    if will_intersect:
        translation = min_distance * translation_axis
    return intersecting, will_intersect, translation


def confidence_ellipsoid(
    data: dict[str, np.ndarray],
    lambda_: float = 1e-5,
    delta: float = 0.1,
    sigma: float = 0.1,
    param_bound: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a confidence ellipsoid over the parameter theta, where y = theta^T phi

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param lambda_: l2 regularization parameter
    :param delta: confidence level
    :param sigma: noise covariance
    :param param_bound: an upper-bound on the parameter norm
    :return: estimated theta, Gramian matrix G_N_lambda, radius beta_N
    """
    phi = np.array(data["features"])
    y = np.array(data["outputs"])
    g_n_lambda = 1 / sigma * np.transpose(phi) @ phi + lambda_ * np.identity(
        phi.shape[-1]
    )
    theta_n_lambda = np.linalg.inv(g_n_lambda) @ np.transpose(phi) @ y / sigma
    d = theta_n_lambda.shape[0]
    beta_n = (
        np.sqrt(2 * np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_**d) / delta))
        + np.sqrt(lambda_ * d) * param_bound
    )
    return theta_n_lambda, g_n_lambda, beta_n


def confidence_polytope(
    data: dict, parameter_box: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute a confidence polytope over the parameter theta, where y = theta^T phi

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: a box [theta_min, theta_max]  containing the parameter theta
    :return: estimated theta, polytope vertices, Gramian matrix G_N_lambda, radius beta_N
    """
    param_bound = np.amax(np.abs(parameter_box))
    theta_n_lambda, g_n_lambda, beta_n = confidence_ellipsoid(
        data, param_bound=param_bound
    )

    values, pp = np.linalg.eig(g_n_lambda)
    radius_matrix = np.sqrt(beta_n) * np.linalg.inv(pp) @ np.diag(np.sqrt(1 / values))
    h = np.array(list(itertools.product([-1, 1], repeat=theta_n_lambda.shape[0])))
    d_theta = np.array([radius_matrix @ h_k for h_k in h])

    # Clip the parameter and confidence region within the prior parameter box.
    theta_n_lambda = np.clip(theta_n_lambda, parameter_box[0], parameter_box[1])
    for k, _ in enumerate(d_theta):
        d_theta[k] = np.clip(
            d_theta[k],
            parameter_box[0] - theta_n_lambda,
            parameter_box[1] - theta_n_lambda,
        )
    return theta_n_lambda, d_theta, g_n_lambda, beta_n


def is_valid_observation(
    y: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    gramian: np.ndarray,
    beta: float,
    sigma: float = 0.1,
) -> bool:
    """
    Check if a new observation (phi, y) is valid according to a confidence ellipsoid on theta.

    :param y: observation
    :param phi: feature
    :param theta: estimated parameter
    :param gramian: Gramian matrix
    :param beta: ellipsoid radius
    :param sigma: noise covariance
    :return: validity of the observation
    """
    y_hat = np.tensordot(theta, phi, axes=[0, 0])
    error = np.linalg.norm(y - y_hat)
    eig_phi, _ = np.linalg.eig(phi.transpose() @ phi)
    eig_g, _ = np.linalg.eig(gramian)
    error_bound = np.sqrt(np.amax(eig_phi) / np.amin(eig_g)) * beta + sigma
    return error < error_bound


def is_consistent_dataset(data: dict, parameter_box: np.ndarray = None) -> bool:
    """
    Check whether a dataset {phi_n, y_n} is consistent

    The last observation should be in the confidence ellipsoid obtained by the N-1 first observations.

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: a box [theta_min, theta_max]  containing the parameter theta
    :return: consistency of the dataset
    """
    train_set = copy.deepcopy(data)
    y, phi = train_set["outputs"].pop(-1), train_set["features"].pop(-1)
    y, phi = np.array(y)[..., np.newaxis], np.array(phi)[..., np.newaxis]
    if train_set["outputs"] and train_set["features"]:
        theta, _, gramian, beta = confidence_polytope(
            train_set, parameter_box=parameter_box
        )
        return is_valid_observation(y, phi, theta, gramian, beta)
    else:
        return True


def near_split(x, num_bins=None, size_bins=None):
    """
    Split a number into several bins with near-even distribution.

    You can either set the number of bins, or their size.
    The sum of bins always equals the total.
    :param x: number to split
    :param num_bins: number of bins
    :param size_bins: size of bins
    :return: list of bin sizes
    """
    if num_bins:
        quotient, remainder = divmod(x, num_bins)
        return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)
    elif size_bins:
        return near_split(x, num_bins=int(np.ceil(x / size_bins)))


def distance_to_circle(center, radius, direction):
    scaling = radius * np.ones((2, 1))
    a = np.linalg.norm(direction / scaling) ** 2
    b = -2 * np.dot(np.transpose(center), direction / np.square(scaling))
    c = np.linalg.norm(center / scaling) ** 2 - 1
    root_inf, root_sup = solve_trinom(a, b, c)
    if root_inf and root_inf > 0:
        distance = root_inf
    elif root_sup and root_sup > 0:
        distance = 0
    else:
        distance = np.infty
    return distance


def distance_to_rect(line: tuple[np.ndarray, np.ndarray], rect: list[np.ndarray]):
    """
    Compute the intersection between a line segment and a rectangle.

    See https://math.stackexchange.com/a/2788041.
    :param line: a line segment [R, Q]
    :param rect: a rectangle [A, B, C, D]
    :return: the distance between R and the intersection of the segment RQ with the rectangle ABCD
    """
    r, q = line
    a, b, c, d = rect
    u = b - a
    v = d - a
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    rqu = (q - r) @ u
    rqv = (q - r) @ v
    interval_1 = [(a - r) @ u / rqu, (b - r) @ u / rqu]
    interval_2 = [(a - r) @ v / rqv, (d - r) @ v / rqv]
    interval_1 = interval_1 if rqu >= 0 else list(reversed(interval_1))
    interval_2 = interval_2 if rqv >= 0 else list(reversed(interval_2))
    if (
        interval_distance(*interval_1, *interval_2) <= 0
        and interval_distance(0, 1, *interval_1) <= 0
        and interval_distance(0, 1, *interval_2) <= 0
    ):
        return max(interval_1[0], interval_2[0]) * np.linalg.norm(q - r)
    else:
        return np.inf


def solve_trinom(a, b, c):
    delta = b**2 - 4 * a * c
    if delta >= 0:
        return (-b - np.sqrt(delta)) / (2 * a), (-b + np.sqrt(delta)) / (2 * a)
    else:
        return None, None


def load_lanelet_map(laneletmap):
    points = dict()
    way = dict()
    lanes = dict()
    for ls in laneletmap.lineStringLayer:

        if "type" not in ls.attributes.keys():
            raise RuntimeError("ID " + str(ls.id) + ": Linestring type must be specified")
        elif ls.attributes["type"] == "curbstone":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "line_thin":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "line_thick":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=2, zorder=10)
        elif ls.attributes["type"] == "pedestrian_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif ls.attributes["type"] == "bike_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif ls.attributes["type"] == "stop_line":
            type_dict = dict(color="white", linewidth=3, zorder=10)
        elif ls.attributes["type"] == "virtual":
            type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
        elif ls.attributes["type"] == "road_border":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "guard_rail":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "traffic_sign":
            continue
        elif ls.attributes["type"] == "building":
            type_dict = dict(color="pink", zorder=1, linewidth=5)
        elif ls.attributes["type"] == "spawnline":
            if ls.attributes["spawn_type"] == "start":
                type_dict = dict(color="green", zorder=11, linewidth=2)
            elif ls.attributes["spawn_type"] == "end":
                type_dict = dict(color="red", zorder=11, linewidth=2)
        else:
            continue
        attr = dict(ls.attributes)
        type_dict.update(attr)
        for p in ls:
            points[p.id] = [p.y, p.x, p.z]
        way_points = [[pt.y, pt.x, pt.z] for pt in ls]
        way[ls.id] = {'type': type_dict, 'points': way_points}

        if ls.attributes["type"] == "pedestrian_marking":
            way[ls.id]['spline'] = Spline2D(np.array(way_points)[:, 0], np.array(way_points)[:, 1])

    for ll in laneletmap.laneletLayer:
        left = [[pt.y, pt.x, pt.z] for pt in ll.leftBound]
        left_point_ids = [pt.id for pt in ll.leftBound]
        leftBound=dict()
        leftBound['id'] = ll.leftBound.id
        leftBound['ids'] = left_point_ids
        leftBound['points'] = left
        leftBound['spline'] = Spline2D(np.array(left)[:, 0], np.array(left)[:, 1])
        leftBound['type'] = way[ll.leftBound.id]['type']
        right = [[pt.y, pt.x, pt.z] for pt in ll.rightBound]
        right_point_ids = [pt.id for pt in ll.rightBound]
        rightBound = dict()
        rightBound['id'] = ll.rightBound.id
        rightBound['ids'] = right_point_ids
        rightBound['points'] = right
        rightBound['spline'] = Spline2D(np.array(right)[:, 0], np.array(right)[:, 1])
        rightBound['type'] = way[ll.rightBound.id]['type']
        center = [[pt.y, pt.x, pt.z] for pt in ll.centerline]

        lanes[ll.id] = {
            'left': leftBound,
            'right': rightBound,
            'center_points': center,
            'spline': Spline2D(np.array(center)[:, 0], np.array(center)[:, 1]),
            'left_lane_id': None,
            'right_lane_id': None,
            'predecessors': set(),
            'successors': set(),
            'is_intersection': leftBound['type']['type'] == 'virtual' or rightBound['type']['type'] == 'virtual',
        }

    lanes_copy = copy.deepcopy(lanes)
    # {30025, 30057}
    # TODO 添加对向车道以逆行
    for k, v in lanes_copy.items():
        end_point = v['center_points'][-1]
        for k1, v1 in lanes_copy.items():
            if k != k1:
                start_point1 = v1['center_points'][0]

                distance = np.linalg.norm(np.array(end_point) - np.array(start_point1))

                if v['left']['id'] == v1['right']['id']:
                    if lanes[k]['spline'].s[-1] < 5 or lanes[k1]['spline'].s[-1] < 5:
                        continue
                    lanes[k]['left_lane_id'] = k1
                    lanes[k1]['right_lane_id'] = k

                if not v['is_intersection'] and v['left']['id'] == v1['left']['id']:
                    if lanes[k]['spline'].s[-1] < 5 or lanes[k1]['spline'].s[-1] < 5:
                        continue
                    lanes[k]['reverse_lane'] = k1
                    lanes[k]['reverse_lane_spline'] = Spline2D(np.array(lanes[k1]['center_points'][::-1])[:, 0], np.array(lanes[k1]['center_points'][::-1])[:, 1])

                if distance < 0.1:
                    lanes[k]['successors'].add(k1)
                    lanes[k1]['predecessors'].add(k)

    # # short_lanes = [k for k, v in lanes.items() if v['spline'].s[-1] < 5]
    # # # TODO 20 22 39修复他们连接关系
    # lanes_copy = copy.deepcopy(lanes)
    # for k, lane in lanes_copy.items():
    #     if lane['spline'].s[-1] < 5:
    #         sucessors = lane['successors']
    #         predecessors = lane['predecessors']
    #         if len(sucessors) == 0:
    #             while len(predecessors) > 0:
    #                 predecessor = predecessors.pop()
    #                 lanes[predecessor]['center_points'] = lanes_copy[predecessor]['center_points'] + lanes_copy[k]['center_points'][1:]
    #                 lanes[predecessor]['spline'] = Spline2D(np.array(lanes[predecessor]['center_points'])[:, 0], np.array(lanes[predecessor]['center_points'])[:, 1])
    #             del lanes[k]
    #         elif len(predecessors) == 0:
    #             while len(sucessors) > 0:
    #                 sucessor = sucessors.pop()
    #                 lanes[sucessor]['center_points'] = lanes_copy[k]['center_points'] + lanes_copy[sucessor]['center_points'][1:]
    #                 lanes[sucessor]['spline'] = Spline2D(np.array(lanes[sucessor]['center_points'])[:, 0], np.array(lanes[sucessor]['center_points'])[:, 1])
    #             del lanes[k]
    #
    # lanes_copy = copy.deepcopy(lanes)
    # for k, lane in lanes_copy.items():
    #     if lane['spline'].s[-1] < 5:
    #         sucessors = lane['successors']
    #         predecessors = lane['predecessors']
    #         if len(predecessors) == 1 and len(sucessors) == 1:
    #             # predecessor = predecessors.pop()
    #             sucessor = sucessors.pop()
    #             if len(lanes[sucessor]['predecessors']) > 1: # 如果后继有多个前驱
    #                 continue
    #             # lanes[predecessor]['center_points'] = lanes[predecessor]['center_points'] + lanes[k]['center_points'][1:]
    #             # lanes[predecessor]['spline'] = Spline2D(np.array(lanes[predecessor]['center_points'])[:, 0], np.array(lanes[predecessor]['center_points'])[:, 1])
    #             lanes[sucessor]['center_points'] = lanes[k]['center_points'] + lanes[sucessor]['center_points'][1:]
    #             lanes[sucessor]['spline'] = Spline2D(np.array(lanes[sucessor]['center_points'])[:, 0], np.array(lanes[sucessor]['center_points'])[:, 1])
    #             del lanes[k]
    #
    # lanes_copy = copy.deepcopy(lanes)
    # for k, lane in lanes_copy.items():
    #     if lane['spline'].s[-1] < 5:
    #         sucessors = lane['successors']
    #         predecessors = lane['predecessors']
    #         if len(predecessors) == 1:
    #
    #             while len(sucessors) > 0:
    #                 sucessor = sucessors.pop()
    #                 lanes[sucessor]['center_points'] = lanes[predecessors.pop()]['center_points'] + lanes[sucessor]['center_points'][1:]
    #                 lanes[sucessor]['spline'] = Spline2D(np.array(lanes[sucessor]['center_points'])[:, 0], np.array(lanes[sucessor]['center_points'])[:, 1])
    #             # predecessor = predecessors.pop()
    #             # lanes[predecessor]['center_points'] = lanes[predecessor]['center_points'] + lanes[k]['center_points'][1:]
    #             # lanes[predecessor]['spline'] = Spline2D(np.array(lanes[predecessor]['center_points'])[:, 0],
    #             #                                         np.array(lanes[predecessor]['center_points'])[:, 1])
    #
    #             del lanes[k]

    # lanes_copy = copy.deepcopy(lanes)
    # for k, lane in lanes_copy.items():
    #     if lane['spline'].s[-1] < 5:
    #         sucessors = lane['successors']
    #         if len(sucessors) == 1:
    #             sucessor = sucessors.pop()
    #             lanes[sucessor]['center_points'] = lanes[k]['center_points'] + lanes[sucessor]['center_points'][1:]
    #             lanes[sucessor]['spline'] = Spline2D(np.array(lanes[sucessor]['center_points'])[:, 0], np.array(lanes[sucessor]['center_points'])[:, 1])
    #             del lanes[k]
    #
    # for k, v in lanes.items():
    #     v['successors'] = set()
    #     v['predecessors'] = set()
    #
    # for k, v in lanes.items():
    #     end_point = v['center_points'][-1]
    #     for k1, v1 in lanes.items():
    #         if k != k1:
    #             start_point1 = v1['center_points'][0]
    #
    #             if np.linalg.norm(np.array(end_point) - np.array(start_point1)) < 0.1:
    #                 lanes[k]['successors'].add(k1)
    #                 lanes[k1]['predecessors'].add(k)

    return {'points': points, 'way': way, 'lanes': lanes}
