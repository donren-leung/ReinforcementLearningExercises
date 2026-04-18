import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

from model_free.blackjack.agents.agent import HIT, STICK

def plot_policy(policy_grids: tuple[np.ndarray, np.ndarray],
                value_grids: tuple[np.ndarray, np.ndarray],
                optimal_pi: bool) -> Figure:
    old_rcParams = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(12, 8))

    # Outer grid: gutter column + content block
    outer = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[0.4, 2.0],
        wspace=0.0,   # no extra gap between gutter and content block
        hspace=0.0,
    )

    # Inner grid only for policy/value columns
    inner = outer[:, 1].subgridspec(
        nrows=2, ncols=2,
        width_ratios=[1.2, 1.8],
        wspace=0.5,  # applies only between policy and value
        hspace=0.3,
    )

    write_gutter_label(fig.add_subplot(outer[0, 0]), "Usable\nace", fontsize=20, x=0.4, y=0.5)
    write_gutter_label(fig.add_subplot(outer[1, 0]), "No\nusable\nace", fontsize=20, x=0.4, y=0.5)

    set1 = plt.get_cmap("Pastel1")
    colours = [set1(0), set1(2)] # STAND=red, HIT=green
    cmap = ListedColormap(colours)

    X, Y = np.meshgrid(range(1, 11), range(12, 22))
    for row_index, (policy_grid, value_grid) in enumerate(zip(policy_grids, value_grids)):
        policy_ax = fig.add_subplot(inner[row_index, 0])
        if row_index == 0:
            if optimal_pi:
                policy_ax.set_title(r"$\pi_*$", fontsize=30, va="bottom")
            else:
                policy_ax.set_title(r"$\pi$", fontsize=30, va="bottom")
        policy_ax.imshow(policy_grid, aspect="auto", vmin=0, vmax=1, cmap=cmap, origin="lower")
        policy_ax.set_xlabel("Dealer showing")
        policy_ax.set_xticks(range(10), labels=["A"] + list(str(i) for i in range(2, 11)))
        
        policy_ax.set_ylabel("Player sum")
        policy_ax.set_yticks(range(11), labels=[str(i) for i in range(11, 22)])
        policy_ax.yaxis.set_label_position("right")
        policy_ax.yaxis.tick_right()

        for i in range(policy_grid.shape[0]):
            for j in range(policy_grid.shape[1]):
                if not np.isnan(policy_grid[i, j]):
                    policy_ax.text(j, i, "H" if policy_grid[i, j] == HIT else "S",
                                   ha="center", va="center", color="black")

        value_ax = fig.add_subplot(inner[row_index, 1], projection="3d")
        if row_index == 0:
            if optimal_pi:
                value_ax.set_title(r"$v_{*}$", fontsize=30, va="bottom")
            else:
                value_ax.set_title(r"$v_{\pi}$", fontsize=30, va="bottom")
        wf = value_ax.plot_wireframe(X, Y, value_grid, rstride=1, cstride=1)
        wf.set_clip_on(False)
        style_blackjack_ax(value_ax, row_index, show_labels=True,
                           zoom=1.6, focal_length=0.5,
                           ax_label_fontsize=12)

    plt.rcParams["mathtext.fontset"] = old_rcParams
    return fig

def plot_value(snapshots: list[tuple[int, tuple[np.ndarray, np.ndarray]]]) -> Figure:
    n_columns = len(snapshots)
    fig = plt.figure(figsize=(6 * n_columns + 2.2, 10))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=n_columns + 1,
        width_ratios=[0.1] + [1] * n_columns,
        height_ratios=[1, 1],
        hspace=-0.26,
        wspace=0.05,
    )

    X, Y = np.meshgrid(range(1, 11), range(12, 22))
    row_labels = ["Usable\nace", "No\nusable\nace"]

    for column_index, (episodes, value_grids) in enumerate(snapshots):
        for row_index, (row_label, grid) in enumerate(zip(row_labels, value_grids)):
            if column_index == 0:
                label_ax = fig.add_subplot(gs[row_index, 0])
                write_gutter_label(label_ax, row_label, fontsize=18, x=0.4, y=0.5)

            ax = fig.add_subplot(gs[row_index, column_index + 1], projection="3d")
            ax.plot_wireframe(X, Y, grid, rstride=1, cstride=1)
            style_blackjack_ax(ax, row_index, show_labels=column_index == n_columns - 1)

            if row_index == 0:
                ax.set_zorder(2)
                bbox = ax.get_position()
            elif row_index == 1:
                ax.set_zorder(1)

    # fig.subplots_adjust(top=0.9, bottom=0.1, left=0.02, right=0.98) 
    fig.canvas.draw()
    for column_index, (episodes, _) in enumerate(snapshots):
        cell = gs[0, column_index + 1].get_position(fig)
        fig.text(
            x=(cell.x0 + cell.x1) / 2,
            y=min(0.80, cell.y1 + 0.02),
            s=f"After {episodes:,} episodes",
            ha="center",
            va="bottom",
            fontsize=20,
        )

    return fig

def write_gutter_label(ax: Axes, text: str, x: float, y: float, fontsize: int=18) -> None:
    ax.set_axis_off()
    ax.text(
        x=x,
        y=y,
        s=text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=999
    )

def style_blackjack_ax(ax, row_idx: int, show_labels: bool,
                       zoom: float=0.95, focal_length: float=0.30,
                       ax_label_fontsize: int=14) -> None:
    ax.set_xlim(1, 10)
    ax.set_ylim(12, 21)
    ax.set_zlim(-1, 1)


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if show_labels and row_idx == 0:
        ax.text(1 - 0.4, 12 - 0.4, -1, "-1", zdir=None, ha="center", va="center", fontsize=ax_label_fontsize)
        ax.text(1 - 0.4, 12 - 0.4, 1, "+1", zdir=None, ha="center", va="center", fontsize=ax_label_fontsize)
    elif show_labels:
        ax.text((10 + 2 - 0.5) / 2, 12 - 0.8, -1, "Dealer showing", zdir="x", ha="center", va="center", fontsize=ax_label_fontsize)
        ax.text(2 - 0.5,            12 - 0.8, -1, "A", zdir="x", ha="center", va="center", fontsize=ax_label_fontsize)
        ax.text(10 - 0.2,           12 - 0.8, -1, "10", zdir="x", ha="center", va="center", fontsize=ax_label_fontsize)

        ax.text(10 + 0.8, (20 + 12) / 2, -1, "Player sum", zdir="y", ha="center", va="center", fontsize=ax_label_fontsize)
        ax.text(10 + 0.8, 12 + 0.5,      -1, "12", zdir="y", ha="center", va="center", fontsize=ax_label_fontsize)
        ax.text(10 + 0.8, 20 + 0.5,      -1, "21", zdir="y", ha="center", va="center", fontsize=ax_label_fontsize)

    ax.tick_params(axis="x", pad=-2)
    ax.tick_params(axis="y", pad=-2)

    # remove grid
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0

    # transparent panes; keep no automatic pane edges
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0, 0, 0, 0))

    ax.set_proj_type("persp", focal_length=focal_length)
    ax.view_init(elev=25, azim=-70)

    # flatter z dimension
    ax.set_box_aspect((10, 10, 2.5), zoom=zoom)
    # manually draw full box so front edges always appear
    draw_3d_box(ax, (1, 10), (12, 21), (-1, 1), color="black", linewidth=1)

def draw_3d_box(ax, xlim, ylim, zlim, **kwargs):
    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim

    corners = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        ax.plot(
            [corners[i][0], corners[j][0]],
            [corners[i][1], corners[j][1]],
            [corners[i][2], corners[j][2]],
            clip_on=False,
            **kwargs,
        )
