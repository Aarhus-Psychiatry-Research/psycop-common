import patchworklib as pw
import plotnine as pn
from git import Sequence


def create_patchwork_grid(
    plots: Sequence[pn.ggplot],
    single_plot_dimensions: tuple[float, float],
    n_in_row: int,
    add_subpanels_letters: bool = True,
) -> pw.Bricks:
    """Create a grid from a sequence of ggplto objects."""
    bricks = [pw.load_ggplot(plot, figsize=single_plot_dimensions) for plot in plots]

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    rows = []
    current_row = []

    for i in range(len(bricks)):
        # Add the letter
        if add_subpanels_letters:
            bricks[i].set_index(alphabet[i].upper())

        # Add it to the row
        current_row.append(bricks[i])

        if i % n_in_row != 0:
            # Rows should consist of two elements
            rows.append(pw.stack(current_row, operator="|"))
            current_row = []

    # Combine the rows
    patchwork = pw.stack(rows, operator="/")
    return patchwork
