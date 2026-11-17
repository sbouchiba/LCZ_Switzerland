"""
Cross-city generalization scan (part 2/2): plots
====================================================

Reads cross_city_scan_results.csv (produced by 08a_cross_city_scan.py) and
makes 4 bar charts, one per (regime, split) combination:
  1. OA_train, regime = allcity  -> one bar per city
  2. OA_test,  regime = allcity  -> one bar per city
  3. OA_train, regime = geneva   -> one bar per city
  4. OA_test,  regime = geneva   -> one bar per city

Bars for a city that CONTRIBUTED to that regime's training set (is_home_city
== True) get a black outline + hatch, so "in-sample" fit is visually
distinguished from genuine out-of-sample generalization.
- In the "allcity" plots, every city is in-sample (all contributed), so this
  mainly matters for the "geneva" plots, where only Geneva itself is in-sample
  and every other city bar represents true cross-city transfer.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

df = pd.read_csv("./tables/cross_city_scan_results.csv")

cities = sorted(df["city"].unique())
city_colors = {c: cm.tab10(i % 10) for i, c in enumerate(cities)}

panels = [
    ("allcity", "OA_train", "OA_train -- regime: allcity"),
    ("allcity", "OA_test", "OA_test -- regime: allcity"),
    ("geneva", "OA_train", "OA_train -- regime: geneva"),
    ("geneva", "OA_test", "OA_test -- regime: geneva"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
axes = axes.flatten()

for ax, (regime, metric, title) in zip(axes, panels):
    sub = df[df["regime"] == regime].set_index("city").loc[cities]

    bar_colors = [city_colors[c] for c in cities]
    edge_colors = ["black" if sub.loc[c, "is_home_city"] else "none" for c in cities]
    hatches = ["//" if sub.loc[c, "is_home_city"] else "" for c in cities]
    linewidths = [2 if sub.loc[c, "is_home_city"] else 0 for c in cities]

    bars = ax.bar(cities, sub[metric], color=bar_colors, edgecolor=edge_colors,
                   linewidth=linewidths, hatch=None)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    for i, city in enumerate(cities):
        val = sub.loc[city, metric]
        ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)

    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Overall Accuracy")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

# Legend explaining the hatch/outline
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="white", edgecolor="black", hatch="//", linewidth=2,
          label="City contributed to this regime's training set (in-sample)"),
    Patch(facecolor="white", edgecolor="none", label="Out-of-sample city"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Cross-city generalization: OA by city, training regime, and split", y=1.01, fontsize=14)
plt.tight_layout()
plt.savefig("./images/cross_city_scan_plots.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved: cross_city_scan_plots.png")