from pathlib import Path
from babygrad.data import load_csv


def main():
    data = load_csv(Path("./data/iris.csv"))
    print(data)


if __name__ == "__main__":
    main()
