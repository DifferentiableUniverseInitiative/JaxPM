import matplotlib.pyplot as plt
import numpy as np


def plot_fields(fields_dict, sum_over=None):
    """
    Plots sum projections of 3D fields along different axes,
    slicing only the first `sum_over` elements along each axis.

    Args:
    - fields: list of 3D arrays representing fields to plot
    - names: list of names for each field, used in titles
    - sum_over: number of slices to sum along each axis (default: fields[0].shape[0] // 8)
    """
    sum_over = sum_over or list(fields_dict.values())[0].shape[0] // 8
    nb_rows = len(fields_dict)
    nb_cols = 3
    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(15, 5 * nb_rows))

    def plot_subplots(proj_axis, field, row, title):
        slicing = [slice(None)] * field.ndim
        slicing[proj_axis] = slice(None, sum_over)
        slicing = tuple(slicing)

        # Sum projection over the specified axis and plot
        axes[row, proj_axis].imshow(
            field[slicing].sum(axis=proj_axis) + 1,
            cmap='magma',
            extent=[0, field.shape[proj_axis], 0, field.shape[proj_axis]])
        axes[row, proj_axis].set_xlabel('Mpc/h')
        axes[row, proj_axis].set_ylabel('Mpc/h')
        axes[row, proj_axis].set_title(title)

    # Plot each field across the three axes
    for i, (name, field) in enumerate(fields_dict.items()):
        for proj_axis in range(3):
            plot_subplots(proj_axis, field, i,
                          f"{name} projection {proj_axis}")

    plt.tight_layout()
    plt.show()


def plot_fields_single_projection(fields_dict,
                                  sum_over=None,
                                  project_axis=0,
                                  vmin=None,
                                  vmax=None,
                                  colorbar=False):
    """
    Plots a single projection (along axis 0) of 3D fields in a grid,
    summing over the first `sum_over` elements along the 0-axis, with 4 images per row.

    Args:
    - fields_dict: dictionary where keys are field names and values are 3D arrays
    - sum_over: number of slices to sum along the projection axis (default: fields[0].shape[0] // 8)
    """
    sum_over = sum_over or list(fields_dict.values())[0].shape[0] // 8
    nb_fields = len(fields_dict)
    nb_cols = 4  # Set number of images per row
    nb_rows = (nb_fields + nb_cols - 1) // nb_cols  # Calculate required rows

    fig, axes = plt.subplots(nb_rows,
                             nb_cols,
                             figsize=(5 * nb_cols, 5 * nb_rows))
    axes = np.atleast_2d(axes)  # Ensure axes is always a 2D array

    for i, (name, field) in enumerate(fields_dict.items()):
        row, col = divmod(i, nb_cols)

        # Define the slice for the 0-axis projection
        slicing = [slice(None)] * field.ndim
        slicing[project_axis] = slice(None, sum_over)
        slicing = tuple(slicing)

        # Sum projection over axis 0 and plot
        a = axes[row,
                 col].imshow(field[slicing].sum(axis=project_axis) + 1,
                             cmap='magma',
                             extent=[0, field.shape[1], 0, field.shape[2]],
                             vmin=vmin,
                             vmax=vmax)
        axes[row, col].set_xlabel('Mpc/h')
        axes[row, col].set_ylabel('Mpc/h')
        axes[row, col].set_title(f"{name} projection 0")
        if colorbar:
            fig.colorbar(a, ax=axes[row, col], shrink=0.7)

    # Remove any empty subplots
    for j in range(i + 1, nb_rows * nb_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()


def stack_slices(array):
    """
    Stacks 2D slices of an array into a single array based on provided partition dimensions.

    Args:
    - array_slices: a 2D list of array slices (list of lists format) where
      array_slices[i][j] is the slice located at row i, column j in the grid.
    - pdims: a tuple representing the grid dimensions (rows, columns).

    Returns:
    - A single array constructed by stacking the slices.
    """
    # Initialize an empty list to store the vertically stacked rows
    pdims = array.sharding.mesh.devices.shape

    field_slices = []

    # Iterate over rows in pdims[0]
    for i in range(pdims[0]):
        row_slices = []

        # Iterate over columns in pdims[1]
        for j in range(pdims[1]):
            slice_index = i * pdims[0] + j
            row_slices.append(array.addressable_data(slice_index))
        # Stack the current row of slices vertically
        stacked_row = np.hstack(row_slices)
        field_slices.append(stacked_row)

    # Stack all rows horizontally to form the full array
    full_array = np.vstack(field_slices)

    return full_array
