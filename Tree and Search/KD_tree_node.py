"""KDT Node class"""


class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis  # Splitting position(x?y?)
        self.value = value  # Splitting index(which)
        self.left = left
        self.right = right
        self.point_indices = point_indices  # points in one area

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False


"""KDT Node class"""
