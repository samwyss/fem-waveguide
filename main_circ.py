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
    mesh = Mesh("./meshes/circle.inp")
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
    a = 0.02
    
    frequencies_s = np.linspace(0.5e9, 2e9, 10)
    k0_s = 2 * np.pi * frequencies_s * np.sqrt(mu * epsilon)
    k0_s_a = k0_s * a

    fig, ax = plt.subplots()
    
    ax.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TE[1])) / k0_s, "b1", label=r"$TE_{11}$")
    ax.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TE[3])) / k0_s, "r3", label=r"$TE_{21}$")
    ax.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TE[5])) / k0_s, "g2", label=r"$TE_{01}$")

    ax.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TM[0])) / k0_s, "c4", label=r"$TM_{01}$")
    ax.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TM[1])) / k0_s, "m+", label=r"$TM_{11}$")
    ax.plot(k0_s_a, np.sqrt((2*np.pi*frequencies_s)**2 * mu * epsilon - np.sqrt(first_eig_vals_TM[3])) / k0_s, "yx", label=r"$TM_{21}$")
    
    ax.set_ylim((0, 1))
    ax.set_xlabel(r"$k_0a$")
    ax.set_ylabel(r"$k_z/k_0$")

    ax.legend(loc="lower right", frameon=False)
    plt.minorticks_on()
    ax.tick_params(
        which="both",
        axis="both",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )

    plt.show()
    
    
    fig, ax = plt.subplots()
    fig.dpi = 300
    fig.set_size_inches(7, 11)

    ax.set_xlabel(r"$\hat{x}$-Position [m]")
    ax.set_ylabel(r"$\hat{y}$-Position [m]")
    plt.minorticks_on()
    ax.tick_params(
        which="both",
        axis="both",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )
    
    img = ax.tripcolor(
        locations[:, 0],
        locations[:, 1],
        np.real(eig_vec_TE[:, 211]),
        cmap="coolwarm",
    )
    
    cbar = fig.colorbar(img, orientation="horizontal", pad=0.1)
    cbar.set_label(r"$H_z$ [A/m]")
    
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
