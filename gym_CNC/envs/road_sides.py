import numpy as np


# 计算单位法线向量
def unit_normals(points):
    vectors = np.diff(points, axis=0)
    normals = np.empty_like(vectors)
    normals[:, 0] = -vectors[:, 1]
    normals[:, 1] = vectors[:, 0]
    norm = np.linalg.norm(normals, axis=1).reshape(-1, 1)
    unit_normals = normals / norm
    return unit_normals


# 计算角平分线向量
def bisector_vectors(normals):
    bisectors = normals[:-1] + normals[1:]
    norm = np.linalg.norm(bisectors, axis=1).reshape(-1, 1)
    unit_bisectors = bisectors / norm
    return unit_bisectors


# 计算偏移距离调整因子
def offset_factors(points):
    vectors = np.diff(points, axis=0)
    cosines = np.einsum("ij,ij->i", vectors[:-1], vectors[1:])
    norms = np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(vectors[1:], axis=1)
    cosines = cosines / norms
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    factors = 1 / np.cos(angles / 2)
    return factors


# 计算边缘点
def edge_points(center_points, width):
    normals = unit_normals(center_points)
    bisectors = bisector_vectors(normals)
    factors = offset_factors(center_points)
    offsets = bisectors * (width * factors).reshape(-1, 1)
    left_edge = center_points[1:-1] + offsets
    right_edge = center_points[1:-1] - offsets
    return left_edge, right_edge