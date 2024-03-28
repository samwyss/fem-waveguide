from src.Mesh import Mesh
from numpy import zeros
from scipy.linalg import eig


class Solver:
    """
    A concrete solver class that assembles and solves 2 sets of FEM matrices for both TE and TM Modes
    """

    def __init__(self, mesh: Mesh):
        """
        Solver constructor
        :param mesh: an instance of a Mesh object
        """
        self.mesh = mesh
        self.assemble_te_matrices()
        self.assemble_tm_matrices()

    def assemble_te_matrices(self):
        # determine the number of elements (excluding elements on the boundary as they are not included in TM problem)
        num_nodes = len(self.mesh.node_location_list)

        a_te = zeros((num_nodes, num_nodes))
        b_te = zeros((num_nodes, num_nodes))

        # assemble matrices
        for global_element_idx, _ in enumerate(self.mesh.connectivity_list):
            for l_idx in range(3):
                for k_idx in range(3):
                    # determine i, j node numbers as indices in matrices
                    i_gl_idx = self.mesh.connectivity_list[global_element_idx][l_idx]
                    j_gl_idx = self.mesh.connectivity_list[global_element_idx][k_idx]

                    # delta function contribution
                    equal_term = 1 if l_idx == k_idx else 0

                    # accumulate values in matrices
                    a_te[i_gl_idx, j_gl_idx] += (
                        1
                        / (4 * self.calc_element_area(global_element_idx))
                        * (
                            self.calc_b_node(global_element_idx, l_idx)
                            * self.calc_b_node(global_element_idx, k_idx)
                            + self.calc_c_node(global_element_idx, l_idx)
                            * self.calc_c_node(global_element_idx, k_idx)
                        )
                    )

                    b_te[i_gl_idx, j_gl_idx] += (
                        self.calc_element_area(global_element_idx)
                        * (1 + equal_term)
                        / 12
                    )

        # assign on object
        self.a_te = a_te
        self.b_te = b_te

    def assemble_tm_matrices(self):
        # determine the number of elements (excluding elements on the boundary as they are not included in TM problem)
        num_nodes = len(self.mesh.node_location_list) - len(self.mesh.boundary_node_set)

        a_tm = zeros((num_nodes, num_nodes))
        b_tm = zeros((num_nodes, num_nodes))

        # assemble matrices
        for global_element_idx, _ in enumerate(self.mesh.connectivity_list):
            for l_idx in range(3):
                for k_idx in range(3):
                    # determine i, j node numbers as indices in matrices
                    i_gl_idx = self.mesh.connectivity_list[global_element_idx][l_idx]
                    j_gl_idx = self.mesh.connectivity_list[global_element_idx][k_idx]

                    if (self.mesh.is_on_boundary(j_gl_idx) or self.mesh.is_on_boundary(i_gl_idx)):
                        pass
                    else:
                        i_gl_idx = self.mesh.connectivity_list[global_element_idx][l_idx] - sum(i < i_gl_idx for i in self.mesh.boundary_node_set)
                        j_gl_idx = self.mesh.connectivity_list[global_element_idx][k_idx] - sum(i < j_gl_idx for i in self.mesh.boundary_node_set)
                        # delta function contribution
                        equal_term = 1 if l_idx == k_idx else 0

                        # accumulate values in matrices
                        a_tm[i_gl_idx, j_gl_idx] += (
                            1
                            / (4 * self.calc_element_area(global_element_idx))
                            * (
                                self.calc_b_node(global_element_idx, l_idx)
                                * self.calc_b_node(global_element_idx, k_idx)
                                + self.calc_c_node(global_element_idx, l_idx)
                                * self.calc_c_node(global_element_idx, k_idx)
                            )
                        )

                        b_tm[i_gl_idx, j_gl_idx] += (
                            self.calc_element_area(global_element_idx)
                            * (1 + equal_term)
                            / 12
                        )

        # assign on object
        self.a_tm = a_tm
        self.b_tm = b_tm

    def calc_a_node(self, global_element_num: int, local_node_idx: int) -> int:

        # get coordinate list of all nodes in element
        clist = self.mesh.get_coord_list(global_element_num)

        # return intended a value
        if 0 == local_node_idx:
            return clist[1][0] * clist[2][1] - clist[2][0] * clist[1][1]
        if 1 == local_node_idx:
            return clist[2][0] * clist[0][1] - clist[0][0] * clist[2][1]
        if 2 == local_node_idx:
            return clist[0][0] * clist[1][1] - clist[1][0] * clist[0][1]

        # handle base case
        print("invalid local_node_idx")
        return 0

    def calc_b_node(self, global_element_num: int, local_node_idx: int) -> int:

        # get coordinate list of all nodes in element
        clist = self.mesh.get_coord_list(global_element_num)

        # return intended a value
        if 0 == local_node_idx:
            return clist[1][1] - clist[2][1]
        if 1 == local_node_idx:
            return clist[2][1] - clist[0][1]
        if 2 == local_node_idx:
            return clist[0][1] - clist[1][1]

        # handle base case
        print("invalid local_node_idx")
        return 0

    def calc_c_node(self, global_element_num: int, local_node_idx: int) -> int:

        # get coordinate list of all nodes in element
        clist = self.mesh.get_coord_list(global_element_num)

        # return intended a value
        if 0 == local_node_idx:
            return clist[2][0] - clist[1][0]
        if 1 == local_node_idx:
            return clist[0][0] - clist[2][0]
        if 2 == local_node_idx:
            return clist[1][0] - clist[0][0]

        # handle base case
        print("invalid local_node_idx")
        return 0

    def calc_element_area(self, global_element_num: int) -> float:
        return (
            1
            / 2
            * (
                self.calc_b_node(global_element_num, 0)
                * self.calc_c_node(global_element_num, 1)
                - self.calc_b_node(global_element_num, 1)
                * self.calc_c_node(global_element_num, 0)
            )
        )

    def solve_eig_probs(self):

        (eig_values_TE, eig_vecs_TE) = eig(self.a_te, self.b_te)
        (eig_values_TM, eig_vecs_TM) = eig(self.a_tm, self.b_tm)

        return (eig_values_TE, eig_vecs_TE, eig_values_TM, eig_vecs_TM)
