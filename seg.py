
from collections import namedtuple
import numpy as np
from plyfile import PlyData, PlyElement
import copy
import hashlib
from collections import defaultdict

# Define Node and Leaf with range attribute
Node = namedtuple("Node", ["left", "right", "axis", "threshold", "range"])
Leaf = namedtuple("Leaf", ["point", "range","idx"])


def compute_point_cloud_bounds(points):
    """
    Compute the minimum and maximum values for each dimension of a point cloud.

    Args:
    points (np.array): A numpy array of shape (N, 3) where N is the number of points,
                       and each point is represented by its X, Y, Z coordinates.

    Returns:
    list: A list of three lists, each containing [min, max] for X, Y, and Z dimensions respectively.
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    if points.shape[1] != 3:
        raise ValueError("Input points should have 3 dimensions (X, Y, Z)")

    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)

    return [[min_bounds[i], max_bounds[i]] for i in range(3)]

def get_index_and_ranges(root):
    index_and_ranges = []

    def _traverse(node):
        if isinstance(node, Leaf):
            index_and_ranges.append([node.range,node.idx])
        elif isinstance(node, Node):
            _traverse(node.left)
            _traverse(node.right)

    _traverse(root)
    return index_and_ranges

def process_list(input_list):
    result = []
    for sublist in input_list:
        if len(sublist) == 2:
            a, b = sublist
            if b > a + 1:
                result.append(int((a + b) / 2))
            else:
                result.append(a)
        else:
            # 处理子列表不是两个元素的情况
            result.append(sublist[-1])
    return result


def detect_duplicates_hash(data):
    """
    Detect duplicates in a large dataset using the hash method.

    :param data: An iterable of data items
    :return: A dictionary where keys are hashes and values are lists of duplicate items
    """
    hash_dict = defaultdict(list)

    for item in data:
        # Convert item to a string and encode to bytes
        item_str = str(item).encode('utf-8')

        # Create a hash of the item
        item_hash = hashlib.md5(item_str).hexdigest()

        # Add the item to the list for this hash
        hash_dict[item_hash].append(item)

    # Filter out unique items
    duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}

    return duplicates

class PointKDTree:
    def __init__(self, points, point_range, space_range):
        self.points = np.array(points)
        self.space_range = space_range
        self.point_range = point_range
        self.root = self._build(np.arange(len(points)), space_range)
        self.depth = self.get_depth()

    def _build(self, indices, space_range, depth=0):

        if len(indices) == 1:
            point = self.points[indices[0]]
            ranges = process_list(space_range)
            return Leaf(point=point, range=ranges,idx=indices[0])
        axis = depth % self.points.shape[1]
        while space_range[axis][0]+1 == space_range[axis][1]:
            axis = (axis + 1) % self.points.shape[1]

        sorted_indices = indices[np.argsort(self.points[indices, axis])]
        median_idx = len(sorted_indices) // 2

        left_indices = sorted_indices[:median_idx]
        right_indices = sorted_indices[median_idx:]
        threshold = self.points[sorted_indices[median_idx], axis]

        left_range = copy.deepcopy(space_range)
        right_range = copy.deepcopy(space_range)
        # th = int((threshold-self.point_range[axis][0])/(self.point_range[axis][1]-self.point_range[axis][0])*space_range[axis][1])
        # left_range[axis][1] = th
        # right_range[axis][0] = th
        left_range[axis][1] = left_range[axis][0]+int((left_range[axis][1]-left_range[axis][0])/2)
        right_range[axis][0] = right_range[axis][0]+int((right_range[axis][1]-right_range[axis][0])/2)

        return Node(
            left=self._build(left_indices, left_range, depth + 1),
            right=self._build(right_indices, right_range, depth + 1),
            axis=axis,
            threshold=threshold,
            range=space_range
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

#test
# if __name__ == "__main__":
#     data = PlyData.read("point_cloud.ply")
#     points = np.vstack([data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]]).T
#     space_range = [[0,60],[0,160],[0,90]]
#     point_range = compute_point_cloud_bounds(points)
#     print(point_range)
#     tree = PointKDTree(points, point_range, space_range)
#     # print(len(tree.count_and_list_leaves_at_depth(4)))
#     points_and_ranges = get_index_and_ranges(tree.root)
#     for idx, range_ in points_and_ranges:
#         # if range_ == 40460 or range_ == 246716:
#         print(f"Point: {idx}, Range: {range_}")
#     result = detect_duplicates_hash([x[0] for x in points_and_ranges])
#     # print("Duplicates found:")
#     # for hash_value, items in result.items():
#     #     print(f"Hash {hash_value}: {items}")
#
#     # Count total duplicates
#     total_duplicates = sum(len(items) for items in result.values())
#     print(f"Total number of duplicate items: {total_duplicates}")