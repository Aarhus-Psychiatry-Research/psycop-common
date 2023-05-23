from collections.abc import Sequence
from math import ceil

import patchworklib as pw
import plotnine as pn


def create_patchwork_grid(
    plots: Sequence[pn.ggplot],
    single_plot_dimensions: tuple[float, float],
    n_in_row: int,
    add_subpanels_letters: bool = True,
) -> pw.Bricks:
    """Create a grid from a sequence of ggplot objects."""
    print_a4_ratio(plots, single_plot_dimensions, n_in_row)

    bricks = []

    for plot in plots:
        # Iterate here to catch errors while only a single plot is in scope
        # Makes debugging much easier
        bricks.append(pw.load_ggplot(plot, figsize=single_plot_dimensions))

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    rows = []
    current_row = []

    for i in range(len(bricks)):
        # Add the letter
        if add_subpanels_letters:
            bricks[i].set_index(alphabet[i].upper())

        # Add it to the row
        current_row.append(bricks[i])

        row_is_full = i % n_in_row != 0
        all_bricks_used = i == len(bricks) - 1

        if row_is_full or all_bricks_used:
            # Rows should consist of two elements
            rows.append(pw.stack(current_row, operator="|"))
            current_row = []

    # Combine the rows
    patchwork = pw.stack(rows, operator="/")
    return patchwork


def print_a4_ratio(
    plots: Sequence[pn.ggplot],
    single_plot_dimensions: tuple[float, float],
    n_in_row: int,
):
    """Print conversion factor to A4 for a given grid of plots."""
    n_rows = ceil(len(plots) / n_in_row)
    total_height = single_plot_dimensions[1] * n_rows
    total_width = single_plot_dimensions[0] * n_in_row
    a4_ratio = 297 / 210
    new_plot_height_for_a4 = (a4_ratio * total_width - total_height) / n_rows

    print(f"height/width ratio is {total_height/total_width}")
    print(f"Vertical A4 ratio is {a4_ratio}")
    print(
        f"You could decrease single_plot_height to {new_plot_height_for_a4} for the patchwork to fit A4",
    )
