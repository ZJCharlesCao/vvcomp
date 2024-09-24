import numpy as np
from collections import namedtuple
from fd import ForceDirectedPointCloud,min_distance_kdtree
from plyfile import PlyData, PlyElement

# 定义节点类型
Node = namedtuple("Node", ["left", "right", "axis", "threshold"])
Leaf = namedtuple("Leaf", ["point"])


class PointKDTree:
    def __init__(self, points):
        self.points = np.array(points)
        self.root = self._build(np.arange(len(points)))
        self.depth = self.get_depth()

    def _build(self, indices, depth=0):
        if len(indices) == 1:
            return Leaf(self.points[indices[0]])

        axis = depth % self.points.shape[1]
        sorted_indices = indices[np.argsort(self.points[indices, axis])]
        median_idx = len(sorted_indices) // 2

        left_indices = sorted_indices[:median_idx]
        right_indices = sorted_indices[median_idx:]
        threshold = self.points[sorted_indices[median_idx], axis]

        return Node(
            left=self._build(left_indices, depth + 1),
            right=self._build(right_indices, depth + 1),
            axis=axis,
            threshold=threshold
        )

    def get_depth(self):
        return self._get_depth(self.root)

    def _get_depth(self, node):
        if isinstance(node, Leaf):
            return 1
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))

    def count_and_list_leaves_at_depth(self, depth):
        return self._count_and_list_leaves_recursive(self.root, 0, depth)

    def _count_and_list_leaves_recursive(self, node, current_depth, target_depth):
        if current_depth == target_depth:
            return [self._count_and_list_leaves(node.left), self._count_and_list_leaves(node.right)]

        if isinstance(node, Leaf):
            return []

        left_result = self._count_and_list_leaves_recursive(node.left, current_depth + 1, target_depth)
        right_result = self._count_and_list_leaves_recursive(node.right, current_depth + 1, target_depth)
        return left_result + right_result

    def _count_and_list_leaves(self, node):
        if isinstance(node, Leaf):
            return 1, [node.point]
        left_count, left_points = self._count_and_list_leaves(node.left)
        right_count, right_points = self._count_and_list_leaves(node.right)
        return left_count + right_count, left_points + right_points

    def nearest_neighbor(self, query_point):
        best = [None, np.inf]
        self._search(self.root, np.array(query_point), best)
        return best[0], best[1]

    def _search(self, node, query_point, best):
        if isinstance(node, Leaf):
            dist = np.sum((node.point - query_point) ** 2)
            if dist < best[1]:
                best[0] = node.point
                best[1] = dist
            return

        if query_point[node.axis] <= node.threshold:
            self._search(node.left, query_point, best)
            if query_point[node.axis] + np.sqrt(best[1]) > node.threshold:
                self._search(node.right, query_point, best)
        else:
            self._search(node.right, query_point, best)
            if query_point[node.axis] - np.sqrt(best[1]) <= node.threshold:
                self._search(node.left, query_point, best)

# def fd_transform(points, iterations=10):
#     points = np.array(points)
#     tree = PointKDTree(points)
#     depth = tree.depth
#     results = tree.count_and_list_leaves_at_depth(depth-8)
#     original_points = np.empty((0, 3))
#     transform_points = np.empty((0, 3))
#     for count, o_points in results:
#         o_points = np.array(o_points)
#         original_points = np.vstack([original_points, o_points])
#         reconstructor = ForceDirectedPointCloud(o_points)
#         reconstructor.update(iterations=iterations)
#         transform_points = np.vstack([transform_points, reconstructor.points])
#     return transform_points, original_points
#
#
# def float_to_int_mapping(points):
#     voxelsize = min_distance_kdtree(points)
#     scaled_points = np.round(points / voxelsize).astype(int)
#     unique_points, inverse, counts = np.unique(scaled_points, axis=0, return_inverse=True, return_counts=True)
#     if len(unique_points) < len(scaled_points):
#         print(f"警告：检测到 {len(scaled_points) - len(unique_points)} 个重复点。正在调整...")
#
#         for i in range(len(counts)):
#             if counts[i] > 1:
#                 duplicates = np.where(inverse == i)[0]
#                 for j, idx in enumerate(duplicates[1:]):
#                     scaled_points[idx] += np.array([j + 1, j + 1, j + 1])
#
#     return scaled_points

if __name__ == "__main__":
    data = PlyData.read("point_cloud.ply")
    points = np.vstack([data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]).T
    tree = PointKDTree(points)
    print(len(tree.count_and_list_leaves_at_depth(4)))
