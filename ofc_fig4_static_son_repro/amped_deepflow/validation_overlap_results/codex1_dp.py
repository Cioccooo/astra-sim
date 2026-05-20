from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 如果你在 Jupyter / VS Code interactive 里用，可以取消下面这行注释
# %matplotlib inline

# =========================
# 0) Behaviour switches
# =========================
SHOW_PLOTS = True
SAVE_PLOTS = False

PROJECT_ROOT = Path("/Users/dfx/Python/LLM_analytical_tools")
OUTPUT_DIR = PROJECT_ROOT / "amped_deepflow" / "validation_overlap_results" / "overlap_final_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1) Frozen aggregated data
# =========================
whole_training_exposed = pd.DataFrame([
    ["GPT3 175B", "lump\n10 us",         147917.11492919922,  768.0633001327515, 37171.29455733299,  7323.220424652099,  17.166137695312504, 193196.85934901237],
    ["GPT3 175B", "layerwise\n10 us",    147917.11492919922,  768.0633001327515, 37171.29455733299,    76.28354609023525,  0.0,               185932.7563327552],
    ["GPT3 175B", "bucketed(4L)\n10 us", 147917.11492919922,  768.0633001327515, 37171.29455733299,   304.91102457051545,  0.0,               186161.3838112355],
    ["GPT3 175B", "lump\n500 us",        148337.68530273438,  770.2537708282471, 37276.98476839065,  7323.220424652099, 858.306884765625,    194566.451151371],
    ["GPT3 175B", "layerwise\n500 us",   148337.68530273438,  770.2537708282471, 37276.98476839065,    76.28354609023525,  0.0,               186461.20738804352],
    ["GPT3 175B", "bucketed(4L)\n500 us",148337.68530273438,  770.2537708282471, 37276.98476839065,   304.91102457051545,  0.0,               186689.8348665238],

    ["Megatron 530B", "lump\n10 us",         201863.23334270055,  960.0633001327515, 50705.824160708326, 21049.81905619303,   18.77546310424805, 274597.7153228389],
    ["Megatron 530B", "layerwise\n10 us",    201863.23334270055,  960.0633001327515, 50705.824160708326,   200.47446720185525, 0.0,               253729.59527074348],
    ["Megatron 530B", "bucketed(4L)\n10 us", 201863.23334270055,  960.0633001327515, 50705.824160708326,  5315.938840583984,  0.0,               258845.0596441256],
    ["Megatron 530B", "lump\n500 us",        202323.2321887546,   962.2537708282471, 50821.371489895726, 21049.81905619303,  938.7731552124023,  276095.44966088404],
    ["Megatron 530B", "layerwise\n500 us",   202323.2321887546,   962.2537708282471, 50821.371489895726,   200.4744672018541,  0.0,               254307.33191668044],
    ["Megatron 530B", "bucketed(4L)\n500 us",202323.2321887546,   962.2537708282471, 50821.371489895726,  5315.938840584,     0.0,               259422.7962900626],
], columns=[
    "workload",
    "run_setting",
    "tp_comm_s",
    "pp_comm_s",
    "other_comm_s",
    "dp_exposed_comm_s",
    "dp_exposed_reconf_s",
    "exposed_total_bar_s",
])

dp_delta = pd.DataFrame([
    ["GPT3 175B", "lump\n10 us",         7323.220424652099,  7323.220424652099,  17.166137695312504, 17.166137695312504, 0.0,               0.0,               7340.386562347411, 7340.386562347411, 0.0],
    ["GPT3 175B", "layerwise\n10 us",    7323.220424652093,    76.28354609023525, 0.1788139343261719, 0.0,               7246.936878561858, 0.1788139343261719, 7323.399238586419,   76.28354609023525, 7247.115692496184],
    ["GPT3 175B", "bucketed(4L)\n10 us", 7317.864589691164,   304.91102457051545, 0.1788139343261719, 0.0,               7012.953565120649, 0.1788139343261719, 7318.04340362549,   304.91102457051545, 7013.132379054975],
    ["GPT3 175B", "lump\n500 us",        7323.220424652099,  7323.220424652099, 858.306884765625,    858.306884765625,  0.0,               0.0,               8181.527309417724, 8181.527309417724, 0.0],
    ["GPT3 175B", "layerwise\n500 us",   7323.220424652093,    76.28354609023525, 8.940696716308594,  0.0,               7246.936878561858, 8.940696716308594,  7332.161121368402,   76.28354609023525, 7255.877575278167],
    ["GPT3 175B", "bucketed(4L)\n500 us",7317.864589691164,   304.91102457051545, 8.940696716308594,  0.0,               7012.953565120649, 8.940696716308594,  7326.8052864074725, 304.91102457051545, 7021.894261836957],

    ["Megatron 530B", "lump\n10 us",         21049.81905619303, 21049.81905619303,  18.77546310424805, 18.77546310424805, 0.0,               0.0,               21068.59451929728,  21068.59451929728, 0.0],
    ["Megatron 530B", "layerwise\n10 us",    21049.819056193046, 200.47446720185525, 0.1788139343261719, 0.0,             20849.34458899119,  0.1788139343261719, 21049.997870127372,   200.47446720185525, 20849.523402925515],
    ["Megatron 530B", "bucketed(4L)\n10 us", 21044.016901652016, 5315.938840583984,  0.1788139343261719, 0.0,             15728.078061068032, 0.1788139343261719, 21044.195715586342,  5315.938840583984, 15728.256875002358],
    ["Megatron 530B", "lump\n500 us",        21049.81905619303, 21049.81905619303,  938.7731552124023,  938.7731552124023, 0.0,              0.0,               21988.592211405434, 21988.592211405434, 0.0],
    ["Megatron 530B", "layerwise\n500 us",   21049.819056193046, 200.4744672018541,  8.940696716308594, 0.0,             20849.344588991193, 8.940696716308594, 21058.759752909355,   200.4744672018541, 20858.2852857075],
    ["Megatron 530B", "bucketed(4L)\n500 us",21044.016901652016, 5315.938840584,     8.940696716308594, 0.0,             15728.078061068016, 8.940696716308594, 21052.957598368324,  5315.938840584, 15737.018757784324],
], columns=[
    "workload",
    "run_setting",
    "dp_raw_comm_s",
    "dp_exposed_comm_s",
    "dp_raw_reconf_s",
    "dp_exposed_reconf_s",
    "dp_comm_reduction_s",
    "dp_reconf_reduction_s",
    "dp_total_raw_s",
    "dp_total_exposed_s",
    "dp_total_reduction_s",
])


# =========================
# 2) Save local copies of the tables
# =========================
whole_training_exposed.to_csv(OUTPUT_DIR / "whole_training_exposed_by_component.csv", index=False)
dp_delta.to_csv(OUTPUT_DIR / "dp_raw_vs_exposed_delta.csv", index=False)


# =========================
# 3) Plot settings
# =========================
plt.rcParams["figure.dpi"] = 160
plt.rcParams["font.size"] = 11

# 更淡一点的配色
COMPONENT_COLORS = {
    "TP communication": "#9DBBDD",
    "PP communication": "#F6C98F",
    "Other communication": "#A9D39E",
    "DP communication": "#9EC5FE",
    "DP reconfiguration": "#D8B4E2",
    "RAW outline": "#6B7280",
    "EXPOSED outline": "#374151",
}


def add_total_labels(ax, x_positions, totals, y_padding_ratio=0.008, fontsize=9):
    y_pad = max(totals) * y_padding_ratio
    for x, total in zip(x_positions, totals):
        ax.text(x, total + y_pad, f"{total:.0f}", ha="center", va="bottom", fontsize=fontsize)


def finish_figure(fig, filename=None):
    fig.tight_layout()
    if SAVE_PLOTS and filename is not None:
        fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# =========================
# 4) Whole-training RAW vs EXPOSED stacked bars
# =========================
def plot_whole_training_raw_vs_exposed(workload_name: str, output_name: str | None = None):
    df_exp = whole_training_exposed[
        (whole_training_exposed["workload"] == workload_name)
        & (whole_training_exposed["run_setting"].str.contains("10 us", regex=False))
    ].copy()

    df_raw = df_exp.merge(
        dp_delta[["workload", "run_setting", "dp_raw_comm_s", "dp_raw_reconf_s"]],
        on=["workload", "run_setting"],
        how="left",
    )

    df_raw["raw_total_bar_s"] = (
        df_raw["tp_comm_s"]
        + df_raw["pp_comm_s"]
        + df_raw["other_comm_s"]
        + df_raw["dp_raw_comm_s"]
        + df_raw["dp_raw_reconf_s"]
    )

    # x labels 简化，只保留 mode
    display_labels = ["lump", "layerwise", "bucketed(4L)"]

    x = np.arange(len(df_raw))
    width = 0.32

    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    # -------- RAW bars --------
    raw_bottom = np.zeros(len(df_raw))
    raw_components = [
        ("TP communication", df_raw["tp_comm_s"].values),
        ("PP communication", df_raw["pp_comm_s"].values),
        ("Other communication", df_raw["other_comm_s"].values),
        ("DP communication", df_raw["dp_raw_comm_s"].values),
        ("DP reconfiguration", df_raw["dp_raw_reconf_s"].values),
    ]

    for label, values in raw_components:
        ax.bar(
            x - width / 2,
            values,
            width=width,
            bottom=raw_bottom,
            color=COMPONENT_COLORS[label],
            edgecolor=COMPONENT_COLORS["RAW outline"],
            linewidth=0.8,
            alpha=0.75,
            label=label if label not in ax.get_legend_handles_labels()[1] else None,
        )
        raw_bottom += values

    # -------- EXPOSED bars --------
    exp_bottom = np.zeros(len(df_exp))
    exp_components = [
        ("TP communication", df_exp["tp_comm_s"].values),
        ("PP communication", df_exp["pp_comm_s"].values),
        ("Other communication", df_exp["other_comm_s"].values),
        ("DP communication", df_exp["dp_exposed_comm_s"].values),
        ("DP reconfiguration", df_exp["dp_exposed_reconf_s"].values),
    ]

    for label, values in exp_components:
        ax.bar(
            x + width / 2,
            values,
            width=width,
            bottom=exp_bottom,
            color=COMPONENT_COLORS[label],
            edgecolor=COMPONENT_COLORS["EXPOSED outline"],
            linewidth=1.0,
            alpha=1.0,
        )
        exp_bottom += values

    add_total_labels(ax, x - width / 2, df_raw["raw_total_bar_s"].values, y_padding_ratio=0.008)
    add_total_labels(ax, x + width / 2, df_exp["exposed_total_bar_s"].values, y_padding_ratio=0.008)

    # 主 x 轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)

    # 在每对柱子下方单独标 RAW / EXPOSED，避免重叠
    for xi in x:
        ax.text(
            xi - width / 2,
            -0.045,
            "RAW",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=COMPONENT_COLORS["RAW outline"],
        )
        ax.text(
            xi + width / 2,
            -0.045,
            "EXPOSED",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=COMPONENT_COLORS["EXPOSED outline"],
        )

    ax.set_title(f"{workload_name}: whole-training communication, RAW vs EXPOSED @ 10 us")
    ax.set_xlabel("DP overlap mode")
    ax.set_ylabel("Whole-training communication time (s)")
    ax.grid(axis="y", alpha=0.25)

    ax.legend(
        title="Legend",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(bottom=0.16, right=0.80)

    finish_figure(fig, output_name)


# =========================
# 5) Figure B:
#    DP RAW vs EXPOSED
# =========================
def plot_dp_raw_vs_exposed(workload_name: str, output_name: str | None = None):
    df = dp_delta[dp_delta["workload"] == workload_name].copy()

    x = np.arange(len(df))
    width = 0.34

    fig, ax = plt.subplots(figsize=(13.5, 6.8))

    raw_comm = df["dp_raw_comm_s"].values
    raw_reconf = df["dp_raw_reconf_s"].values
    exp_comm = df["dp_exposed_comm_s"].values
    exp_reconf = df["dp_exposed_reconf_s"].values

    # RAW bars
    ax.bar(
        x - width / 2,
        raw_comm,
        width=width,
        label="DP raw communication",
        color="#AFCBFF",
        edgecolor=COMPONENT_COLORS["RAW outline"],
        linewidth=0.8,
    )
    ax.bar(
        x - width / 2,
        raw_reconf,
        width=width,
        bottom=raw_comm,
        label="DP raw reconfiguration",
        color="#E7C8EE",
        edgecolor=COMPONENT_COLORS["RAW outline"],
        linewidth=0.8,
    )

    # EXPOSED bars
    ax.bar(
        x + width / 2,
        exp_comm,
        width=width,
        label="DP exposed communication",
        color="#7FB3FF",
        edgecolor=COMPONENT_COLORS["EXPOSED outline"],
        linewidth=1.0,
    )
    ax.bar(
        x + width / 2,
        exp_reconf,
        width=width,
        bottom=exp_comm,
        label="DP exposed reconfiguration",
        color="#D59BE5",
        edgecolor=COMPONENT_COLORS["EXPOSED outline"],
        linewidth=1.0,
    )

    raw_totals = df["dp_total_raw_s"].values
    exp_totals = df["dp_total_exposed_s"].values

    add_total_labels(ax, x - width / 2, raw_totals, y_padding_ratio=0.012)
    add_total_labels(ax, x + width / 2, exp_totals, y_padding_ratio=0.012)

    ax.set_xticks(x)
    ax.set_xticklabels(df["run_setting"].tolist())

    for xi in x:
        ax.text(xi - width / 2, -0.055, "RAW", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=COMPONENT_COLORS["RAW outline"])
        ax.text(xi + width / 2, -0.055, "EXPOSED", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=COMPONENT_COLORS["EXPOSED outline"])

    ax.set_title(f"{workload_name}: DP communication, RAW vs EXPOSED")
    ax.set_xlabel("Run setting")
    ax.set_ylabel("DP communication time (s)")
    ax.grid(axis="y", alpha=0.25)

    ax.legend(
        title="Legend",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(bottom=0.18, right=0.80)

    finish_figure(fig, output_name)


# =========================
# 6) Run all plots
# =========================
plot_whole_training_raw_vs_exposed(
    workload_name="GPT3 175B",
    output_name="GPT3_175B_whole_training_raw_vs_exposed_10us.png",
)

plot_whole_training_raw_vs_exposed(
    workload_name="Megatron 530B",
    output_name="Megatron_530B_whole_training_raw_vs_exposed_10us.png",
)

print("Done.")
print(f"SHOW_PLOTS = {SHOW_PLOTS}")
print(f"SAVE_PLOTS = {SAVE_PLOTS}")
print(f"OUTPUT_DIR = {OUTPUT_DIR}")
if SAVE_PLOTS:
    print("\nGenerated files:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(" -", p.name)