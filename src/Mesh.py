import meshio
import numpy as np


class Mesh:
    """
    A concrete mesh class used to assemble FEM matrices using a  *.inp file using meshio package
    """

    def __init__(self, path: str) -> None:
        """
        Mesh constructor
        :param path: str, path to *.inp file
        """
        self._mesh = meshio.read(path)

    def get_node_location_list(self) -> list:
        """
        returns a list of node locations in 3D space
        :return: list, node location list
        """
        return self._mesh.points

    def get_connectivity_list(self) -> list:
        """
        returns the connectivity dictionary for all elements, assumes all elements are part of a single block with an
        arbitrary name
        :return: list, connectivity list
        """
        return list(self._mesh.cells_dict.values())[0]

    def get_boundary_node_dict(self) -> dict:
        """
        returns a dictionary of all node sets which are assumed to be boundaries
        :return: dict, boundary node dictionary
        """
        return self._mesh.point_sets

    def is_on_boundary(self, node_idx: int) -> bool:
        """
        determines if a given node with global index node_idx is on a simulation boundary
        :param node_idx:
        :return: bool, True if node exists on a boundary and False if not
        """
        boundary_node_dict = self.get_boundary_node_dict()

        return any(
            node_idx in boundary_node_list
            for boundary_node_list in boundary_node_dict.values()
        )

    def distance_between_nodes(self, node_1_idx: int, node_2_idx: int) -> float:
        """
        determines the distance between any two nodes using global node indexes
        :param node_1_idx: int, global node index 1
        :param node_2_idx: int, global node index 2
        :return:
        """
        node_location_list = self.get_node_location_list()

        node_1_loc = node_location_list[node_1_idx]
        node_2_loc = node_location_list[node_2_idx]

        return np.linalg.norm(node_1_loc - node_2_loc)
