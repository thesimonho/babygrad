import matplotlib.pyplot as plt
from babygrad.nn import Trace


def histogram(x: dict[str, Trace], save_path: str):
    fig, ax = plt.subplots(nrows=len(x.items()))
    for i, (k, v) in enumerate(x.items()):
        ax[i].set_title(f"{k} {v.shape}")
        ax[i].hist(v.data, density=True)
    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Saved histogram to {save_path}.")
