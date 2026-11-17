from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DEFAULT_CLASS_NAMES = [
    "Compact MR", "Open MR", "Open LR", "Large LR",
    "Dense Trees", "Low plants", "Bare Rock/Paved", "Water",
]

# Default bar colour per metric
METRIC_COLORS = {"PA": "lavender", "UA": "mistyrose"}


def plot_accuracy_comparison(
    values,
    references,
    metric="PA",
    class_names=DEFAULT_CLASS_NAMES,
    model_label="AE embedding 2022",
    model_color=None,
    title=None,
    figsize=(7, 5),
    save_path=None,
    dpi=600,
    transparent=True,
    show=True,
):
    """
    Bar plot of a per-class accuracy metric (PA or UA) with reference markers.

    Parameters
    ----------
    values : array-like
        Model values for the metric, one per class.
    references : dict
        {label: {"values": array-like, "color": str}} for each reference, e.g.
        {"S2 2022": {"values": pa_s2, "color": "darkorange"},
         "L8 2022": {"values": pa_l8, "color": "crimson"}}
    metric : str
        Metric name, used for the y-label and the default bar colour ("PA", "UA", ...).
    class_names : list[str]
        x tick labels (same length as `values`).
    model_label : str
        Legend label for the bars.
    model_color : str, optional
        Bar colour. Defaults to METRIC_COLORS[metric], or "lightgray" if unknown.
    title : str, optional
        Figure title.
    save_path : str or Path, optional
        If given, the figure is saved there (parent folders are created).
    dpi, transparent : save options.
    show : bool
        Call plt.show() at the end.

    Returns
    -------
    fig, ax
    """
    values = np.asarray(values, dtype=float)
    if len(values) != len(class_names):
        raise ValueError(
            f"`values` has {len(values)} entries but there are {len(class_names)} class names."
        )

    if model_color is None:
        model_color = METRIC_COLORS.get(metric, "lightgray")

    # Convert references once and validate their length
    refs = {}
    for label, cfg in references.items():
        vals = np.asarray(cfg["values"], dtype=float)
        if len(vals) != len(values):
            raise ValueError(f"Reference '{label}' has {len(vals)} entries, expected {len(values)}.")
        refs[label] = (vals, cfg["color"])

    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=figsize)

    # Model bars
    ax.bar(x, values, width=0.6, alpha=0.7, color=model_color, label=model_label)

    # Reference markers
    for label, (vals, color) in refs.items():
        ax.scatter(x, vals, marker="_", s=400, linewidths=2, color=color, label=label)

    # Common y-range with padding
    all_values = np.concatenate([values, *(v for v, _ in refs.values())])
    ymin = max(0, np.nanmin(all_values) - 5)
    ymax = np.nanmax(all_values) + 8
    ax.set_ylim(ymin, ymax)

    # Axes cosmetics
    if title:
        ax.set_title(title, fontsize=12)
    ax.set_ylabel(f"{metric} (%)", fontsize=10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(which="major", axis="y", alpha=0.3)
    ax.grid(which="minor", axis="y", alpha=0.15)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)

    # Annotate model values above the bars
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    # Annotate differences to each reference at the top (label's first word, e.g. "S2")
    for i, ae in enumerate(values):
        txt = "\n".join(
            f"{label.split()[0]}:{ae - vals[i]:+.1f}" for label, (vals, _) in refs.items()
        )
        ax.text(i, ymax - 1, txt, ha="center", va="top", fontsize=7)

    # Legend below the axes
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=1 + len(refs),
        frameon=False,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, transparent=transparent, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax