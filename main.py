from src.Mesh import Mesh


def main() -> None:
    """
    main simulation driver function
    :returns None
    """
    mesh = Mesh("./mesh.inp")
    print(mesh.distance_between_nodes(0, 6))


if __name__ == "__main__":
    main()
