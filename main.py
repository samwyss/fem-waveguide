from src.Mesh import Mesh


def main() -> None:
    """
    main simulation driver function
    :returns None
    """
    mesh = Mesh("./mesh.inp")
    print(mesh.is_on_boundary())


if __name__ == "__main__":
    main()
