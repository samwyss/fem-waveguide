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
    mesh = Mesh("./meshes/ridged_waveguide.inp")
    solver = Solver(mesh)
    (eig_values_TE, eig_vec_TE, eig_values_TM, eig_vec_TM) = solver.solve_eig_probs()

    first_eig_vals_TE = np.sort(eig_values_TE[np.absolute(eig_values_TE) > 0])[:10]
    first_eig_vals_TM = np.sort(eig_values_TM[np.absolute(eig_values_TM) > 0])[:10]
    print(first_eig_vals_TE)
    print(first_eig_vals_TM)
    first_eig_vecs_idx_TE = np.where(np.in1d(eig_values_TE, first_eig_vals_TE))
    first_eig_vecs_idx_TM = np.where(np.in1d(eig_values_TM, first_eig_vals_TM))
    print(first_eig_vecs_idx_TE)
    print(first_eig_vecs_idx_TM)

    # convert locations and eigenvectors to numpy arrays to make them usable
    locations = np.array(mesh.node_location_list)
    connectivity = np.array(mesh.connectivity_list)

    # make plots
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.figsize"] = (4, 4)
    plt.rcParams["figure.dpi"] = 300
    epsilon = 8.8541878128e-12
    mu = 1.25663706212e-6
    a = 0.02286
    b = 0.01016
    frequencies_th = np.linspace(0.5e9, 2e9, 1000)
    frequencies_s = np.linspace(0.5e9, 2e9, 10)
    k0_th = 2 * np.pi * frequencies_th * np.sqrt(mu * epsilon)
    k0_th_a = 2 * np.pi * frequencies_th * np.sqrt(mu * epsilon) * a
    k0_s = 2 * np.pi * frequencies_s * np.sqrt(mu * epsilon)
    k0_s_a = k0_s * a
    te10_th_kc = np.sqrt((np.pi/a)**2)
    te01_th_kc = np.sqrt((np.pi/b)**2)
    te20_th_kc = np.sqrt((2*np.pi/a)**2)
    tm11_th_kc = np.sqrt((np.pi / a) ** 2 + (np.pi / b) ** 2)
    tm21_th_kc = np.sqrt((2 * np.pi / a) ** 2 + (np.pi / b) ** 2)
    tm31_th_kc = np.sqrt((3 * np.pi / a) ** 2 + (1 * np.pi / b) ** 2)

    """
    plt.plot(k0_th_a, np.sqrt((2*np.pi*frequencies_th)**2 * mu * epsilon - te10_th_kc) / k0_th, "b")
    plt.plot(k0_th_a, np.sqrt((2*np.pi*frequencies_th)**2 * mu * epsilon - te01_th_kc) / k0_th, "g")
    plt.plot(k0_th_a, np.sqrt((2*np.pi*frequencies_th)**2 * mu * epsilon - te20_th_kc) / k0_th, "r")
    plt.plot(k0_th_a, np.sqrt((2*np.pi*frequencies_th)**2 * mu * epsilon - tm11_th_kc) / k0_th, "c")
    plt.plot(k0_th_a, np.sqrt((2*np.pi*frequencies_th)**2 * mu * epsilon - tm21_th_kc) / k0_th, "m")
    plt.plot(k0_th_a, np.sqrt((2*np.pi*frequencies_th)**2 * mu * epsilon - tm31_th_kc) / k0_th, "y")
    """

    plt.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TE[1])) / k0_s, "b1", label=r"$TE_{10}$")
    plt.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TE[2])) / k0_s, "r3", label=r"$TE_{01}$")
    plt.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TE[3])) / k0_s, "g2", label=r"$TE_{20}$")

    plt.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TM[0])) / k0_s, "c4", label=r"$TM_{11}$")
    plt.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TM[1])) / k0_s, "m+", label=r"$TM_{21}$")
    plt.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TM[2])) / k0_s, "yx", label=r"$TM_{31}$")

    plt.ylim((0, 1))
    plt.xlabel(r"$k_0a$")
    plt.ylabel(r"$k_z/k_0$")

    plt.legend(loc="lower right", frameon=False)

    plt.show()


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
        np.real(eig_vec_TE[:, 119]),
        cmap="coolwarm",
    )
    plt.show()
    """
