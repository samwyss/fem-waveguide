from src.Mesh import Mesh
from src.Solver import Solver
import numpy as np


def main() -> None:
    """
    main simulation driver function
    :returns None
    """
    mesh = Mesh("./rec_large.inp")
    solver = Solver(mesh)
    solver.solve_eig_probs()


if __name__ == "__main__":
    main()
