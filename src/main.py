import numpy as np

from Mesh import Mesh
from Solver import Solver


def main() -> None:
    """
    main simulation driver function
    :returns None
    """

    # solve system
    mesh = Mesh("./rectangle.inp")
    solver = Solver(mesh)
    (eig_values_TE, eig_vec_TE, eig_values_TM, eig_vec_TM) = solver.solve_eig_probs()

    # print out kc^2 and the indices of their corresponding eigenvectors -----------------------------------------------
    first_eig_vals_TE = np.sort(eig_values_TE[np.absolute(eig_values_TE) > 0])[:10]
    first_eig_vals_TM = np.sort(eig_values_TM[np.absolute(eig_values_TM) > 0])[:10]
    first_eig_vecs_idx_TE = np.where(np.in1d(eig_values_TE, first_eig_vals_TE))
    first_eig_vecs_idx_TM = np.where(np.in1d(eig_values_TM, first_eig_vals_TM))
    print(f"First TE Eigenvalues: {first_eig_vals_TE}")
    print(f"First TM Eigenvalues: {first_eig_vals_TM}")
    print(f"Indices of First TE Eigenvalues: {first_eig_vecs_idx_TE}")
    print(f"Indices of First TM Eigenvalues: {first_eig_vecs_idx_TM}")

    # insert plotting commands here

# call main function
if __name__ == "__main__":
    main()
