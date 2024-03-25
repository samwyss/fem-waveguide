from src.Mesh import Mesh
from numpy import zeros, nonzero, sort, where, in1d, array, absolute, savetxt
from scipy.linalg import eig, eigvals
import matplotlib.pyplot as plt
from numpy.linalg import norm


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
        num_els = len(self.mesh.connectivity_list)

        a_te = zeros((num_els, num_els))
        b_te = zeros((num_els, num_els))

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
        num_els = len(self.mesh.connectivity_list) - len(self.mesh.boundary_node_set)

        a_tm = zeros((num_els, num_els))
        b_tm = zeros((num_els, num_els))

        i_corr = 0 
        j_corr = 0

        # assemble matrices
        for global_element_idx, _ in enumerate(self.mesh.connectivity_list):
            for l_idx in range(3):
                for k_idx in range(3):
                    # determine i, j node numbers as indices in matrices
                    i_gl_idx = self.mesh.connectivity_list[global_element_idx][l_idx]
                    j_gl_idx = self.mesh.connectivity_list[global_element_idx][k_idx]

                    if (self.mesh.is_on_boundary(j_gl_idx) or self.mesh.is_on_boundary(i_gl_idx)):
                        if self.mesh.is_on_boundary(j_gl_idx) and self.mesh.is_on_boundary(i_gl_idx) :
                            i_corr += 1
                            j_corr += 1
                        if self.mesh.is_on_boundary(
                            j_gl_idx
                        ) and not self.mesh.is_on_boundary(i_gl_idx):
                            j_corr += 1
                        else:
                            i_corr += 1
                    else:
                        i_gl_idx = self.mesh.connectivity_list[global_element_idx][l_idx] - i_corr
                        j_gl_idx = self.mesh.connectivity_list[global_element_idx][k_idx] - j_corr
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
        """
        (eig_values, eig_vecs) = eig(self.a_te, self.b_te)

        first_eigs = sort(eig_values[absolute(eig_values) > 0])[:5]

        print(f"First 5 Eigenvalues TE: {first_eigs}")
        """

        (eig_values, eig_vecs) = eig(self.a_tm, self.b_tm)

        first_eigs = sort(eig_values[absolute(eig_values) > 0])[:5]

        print(f"First 5 Eigenvalues TM: {first_eigs}")
        print(max(eig_values))

        """
        # determine indices
        idxs = where(in1d(eig_values, first_eigs))
        print(idxs)

        # convert locations and eigenvectors to numpy arrays to make them usable
        locations = array(self.mesh.node_location_list)
        connectivity = array(self.mesh.connectivity_list)
        fields = array(eig_vecs)

        # make plots
        plt.tripcolor(
            locations[:, 0],
            locations[:, 1],
            connectivity,
            facecolors=fields[:, 89],
            cmap="coolwarm",
        )
        plt.show()
        """
