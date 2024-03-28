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
        self.connectivity_list = list(raw_mesh.cells_dict.values())[0]  #
        self.boundary_node_set = {
            node_idx
            for boundary_list in raw_mesh.point_sets.values()
            for node_idx in boundary_list
        }

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

        return node_idx in self.boundary_node_set

    def get_coord_list(self, global_element_idx):

        # gets global node indexes associated with global node node_idx
        (node_1, node_2, node_3) = self.connectivity_list[global_element_idx]

        # gets positions of all nodes
        (x1, y1, _) = self.node_location_list[node_1]
        (x2, y2, _) = self.node_location_list[node_2]
        (x3, y3, _) = self.node_location_list[node_3]

        # return list
        return [[x1, y1], [x2, y2], [x3, y3]]
