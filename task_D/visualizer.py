import argparse
import matplotlib.pyplot as plt


def read_input_and_output(infile, outfile):
    # Чтение входа
    input_data = list(map(float, infile.read().split()))
    it = iter(input_data)
    t = int(next(it))
    if t != 1:
        raise ValueError("Визуализатор поддерживает только один тест.")
    n = int(next(it))
    edges = [(int(next(it)) - 1, int(next(it)) - 1) for _ in range(n - 1)]
    points = [(next(it), next(it)) for _ in range(n)]

    # Чтение вывода (перестановка)
    perm = list(map(int, outfile.read().split()))
    if len(perm) != n:
        raise ValueError("Размер перестановки не совпадает с числом вершин")

    assigned = [points[p - 1] for p in perm]
    return n, edges, assigned


def visualize(n, edges, assigned, show_ids=False):
    fig, ax = plt.subplots()
    xs, ys = zip(*assigned)
    ax.scatter(xs, ys, color="blue")

    for i, (x, y) in enumerate(assigned):
        if show_ids:
            ax.text(x, y, str(i + 1), fontsize=8, ha="right", va="bottom")

    for u, v in edges:
        x1, y1 = assigned[u]
        x2, y2 = assigned[v]
        ax.plot([x1, x2], [y1, y2], color="black")

    ax.set_aspect("equal")
    ax.set_title("Дерево на плоскости")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Визуализатор вложенного дерева")
    parser.add_argument(
        "--infile",
        type=argparse.FileType("r"),
        help="Входной файл",
        default="input.txt",
    )
    parser.add_argument(
        "--outfile",
        type=argparse.FileType("r"),
        help="Файл с перестановкой",
        default="output.txt",
    )
    parser.add_argument(
        "--show-ids", action="store_true", help="Показывать номера вершин", default=True
    )
    args, unknown = parser.parse_known_args()

    n, edges, assigned = read_input_and_output(args.infile, args.outfile)
    visualize(n, edges, assigned, args.show_ids)


if __name__ == "__main__":
    main()
