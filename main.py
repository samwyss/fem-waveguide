from src.Mesh import Mesh
from src.Solver import Solver
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    """
    main simulation driver function
    :returns None
    """

    # Rectangular Waveguide ---------------------------------------------
    mesh = Mesh("./meshes/rectangle/rec_large.inp")
    solver = Solver(mesh)
    (eig_values_TE, eig_vec_TE, eig_values_TM, eig_vec_TM) = solver.solve_eig_probs()

    first_eig_vals_TE = np.sort(eig_values_TE[np.absolute(eig_values_TE) > 0])[:10]
    first_eig_vals_TM = np.sort(eig_values_TM[np.absolute(eig_values_TM) > 0])[:10]
    print(first_eig_vals_TE)
    print(first_eig_vals_TM)
    np.savetxt("./data/TE_eigenvalues.csv", np.real(first_eig_vals_TE), delimiter=",", encoding="UTF-8")
    np.savetxt("./data/TM_eigenvalues.csv", np.real(first_eig_vals_TM), delimiter=",", encoding="UTF-8")

    first_eig_vecs_idx_TE = np.where(np.in1d(eig_values_TE, first_eig_vals_TE))
    #print(first_eig_vecs_idx_TE)
    # first_eig_vecs_idx_TM = np.where(np.in1d(eig_values_TM, first_eig_vals_TM))

    # convert locations and eigenvectors to numpy arrays to make them usable
    locations = np.array(mesh.node_location_list)
    connectivity = np.array(mesh.connectivity_list)

    """
    # make plots
    plt.tripcolor(
        locations[:, 0],
        locations[:, 1],
        np.real(eig_vec_TE[119, :]),
        cmap="coolwarm",
    )
    plt.show()
    """

if __name__ == "__main__":
    main()
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
