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

        # load in raw mesh from *.inp file
        raw_mesh = meshio.read(path)

        # assign attributes
        self.node_location_list = raw_mesh.points
        self.connectivity_list = list(raw_mesh.cells_dict.values())[0] #
        self.boundary_nodes = {node_idx for boundary_list in raw_mesh.point_sets.values() for node_idx in boundary_list}

    def __repr__(self) -> str:
        """
        changes object string representation
        :return:
        """
        return f"{self.__dict__}"

    def is_on_boundary(self, node_idx: int) -> bool:
        """
        determines if a given node with global index node_idx is on a simulation boundary
        :param node_idx:
        :return: bool, True if node exists on a boundary and False if not
        """
        return node_idx in self.boundary_nodes

    def distance_between_nodes(self, node_1_idx: int, node_2_idx: int) -> float:
        """
        determines the distance between any two nodes using global node indexes
        :param node_1_idx: int, global node index 1
        :param node_2_idx: int, global node index 2
        :return: float, distance between nodes 1 and 2
        """

        node_1_loc = self.node_location_list[node_1_idx]
        node_2_loc = self.node_location_list[node_2_idx]

        return np.linalg.norm(node_1_loc - node_2_loc)
