import os
import sys

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_and_handle_data(data_name, save_data_as):
    """Loads data from a file path or a Seaborn built-in dataset."""

    if not data_name:
        raise click.UsageError("Data input must be specified using '--data'.")

    df = None
    is_seaborn_dataset = False

    if os.path.exists(data_name):
        click.echo(f"Loading data from file '{data_name}'...")
        try:
            df = pd.read_csv(data_name)
        except Exception as e:
            raise click.ClickException(f"Error loading file '{data_name}': {e}")
    else:
        click.echo(f"File '{data_name}' not found. Attempting to load as a Seaborn dataset...")
        try:
            df = sns.load_dataset(data_name)
            is_seaborn_dataset = True
            click.echo(f"Successfully loaded Seaborn dataset '{data_name}'.")
        except (ValueError, Exception) as e:
            raise click.ClickException(
                f"Could not find file '{data_name}' and it's not a valid Seaborn dataset. Error: {e}"
            )

    # Handle the --save-data-as option (only applicable when a built-in dataset is loaded)
    if is_seaborn_dataset and save_data_as:
        if not os.path.exists(save_data_as):
            click.echo(f"Saving loaded data to '{save_data_as}'...")
            try:
                # Ensure the directory exists
                if dir_name := os.path.dirname(save_data_as):
                    os.makedirs(dir_name, exist_ok=True)
                df.to_csv(save_data_as, index=False)
                click.echo(f"Data saved to '{save_data_as}'.")
            except Exception as e:
                click.echo(f"Error saving data: {e}")

    if df is None:
        raise click.ClickException("Failed to load data.")

    return df


def save_or_show_plot(output_file):
    """Saves the plot to a file if --output is specified, otherwise displays it."""

    if output_file:
        click.echo(f"Saving plot to '{output_file}'...")
        try:
            # Note: plt.gcf() gets the current figure object
            plt.gcf().savefig(output_file)
            click.echo("Save complete.")
        except Exception as e:
            raise click.ClickException(f"Error saving plot: {e}")
    else:
        click.echo("🖥️  --output option not specified. Displaying plot on screen. Close the window to continue.")
        # Note: plt.show() will pause execution until the plot window is closed
        plt.show()


@click.group()
@click.version_option("0.1")
@click.help_option("-h", "--help")
def cli():
    pass


# https://seaborn.pydata.org/generated/seaborn.violinplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save dataset as.")
@click.option("--x", required=True, help="Column name for the X-axis (Required).")
@click.option("--y", required=True, help="Column name for the Y-axis (Required).")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--split/--no-split", default=False, help="Whether to split violins by the hue variable.")
@click.option("--inner", type=click.Choice(["box", "quartile", "point", "stick", None]), default="box")
@click.option("--palette", help="Color palette to use.")
@click.option("--linewidth", type=float, help="Width of the lines that frame the plot elements.")
@click.option("--width", type=float, default=0.8, help="Width of a full element when not using hue nesting.")
@click.option("--cut", type=float, default=2, help="Distance to extend the density past extreme datapoints.")
@click.option(
    "--gridsize", type=int, default=100, help="Number of points in the discrete grid used to evaluate the KDE."
)
@click.option(
    "--bw-method",
    type=click.Choice(["scott", "silverman"]),
    default="scott",
    help="Method for determining the smoothing bandwidth.",
)
@click.option("--bw-adjust", type=float, default=1, help="Factor that scales the bandwidth.")
@click.option("--orient", type=click.Choice(["v", "h"]), help="Orientation of the plot (vertical or horizontal).")
def violinplot(
    input,
    output,
    save_data_as,
    x,
    y,
    hue,
    split,
    inner,
    palette,
    linewidth,
    width,
    cut,
    gridsize,
    bw_method,
    bw_adjust,
    orient,
):
    df = load_and_handle_data(input, save_data_as)
    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "split": split,
        "inner": inner,
        "palette": palette,
        "linewidth": linewidth,
        "width": width,
        "cut": cut,
        "gridsize": gridsize,
        "bw_method": bw_method,
        "bw_adjust": bw_adjust,
        "orient": orient,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.violinplot(data=df, {param_str})")

    plt.figure()
    sns.violinplot(data=df, **plot_params)
    plt.title(f"Violin Plot: {y} by {x}")
    plt.tight_layout()
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.jointplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--x", required=True, help="Column name for the X-axis (Required).")
@click.option("--y", required=True, help="Column name for the Y-axis (Required).")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--kind", type=click.Choice(["scatter", "kde", "hist", "hex", "reg", "resid"]), default="scatter")
@click.option("--height", type=float, default=6, help="Size of the figure (it will be square).")
@click.option("--ratio", type=int, default=5, help="Ratio of joint axes height to marginal axes height.")
@click.option("--space", type=float, default=0.2, help="Space between the joint and marginal axes.")
@click.option("--color", help="Single color for the plot elements.")
@click.option("--palette", help="Color palette to use for the hue variable.")
@click.option("--marginal-ticks/--no-marginal-ticks", default=False, help="Show ticks on the marginal axes.")
def jointplot(data, output, save_data_as, x, y, hue, kind, height, ratio, space, color, palette, marginal_ticks):
    df = load_and_handle_data(data, save_data_as)
    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "kind": kind,
        "height": height,
        "ratio": ratio,
        "space": space,
        "color": color,
        "palette": palette,
        "marginal_ticks": marginal_ticks,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.jointplot(data=df, {param_str})")

    g = sns.jointplot(data=df, **plot_params)
    g.fig.suptitle(f"Joint Plot: {y} vs {x}")
    g.fig.tight_layout()
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.lmplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--x", required=True, help="Column name for the X-axis (Required).")
@click.option("--y", required=True, help="Column name for the Y-axis (Required).")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--col", help="Column name to facet the plot across columns.")
@click.option("--row", help="Column name to facet the plot across rows.")
@click.option("--height", type=float, default=5, help="Height (in inches) of each facet.")
@click.option("--aspect", type=float, default=1, help="Aspect ratio of each facet, so aspect * height gives the width.")
@click.option("--ci", type=int, default=95, help="Size of the confidence interval for the regression estimate.")
@click.option("--scatter/--no-scatter", default=True, help="Draw a scatterplot with the underlying observations.")
@click.option("--fit-reg/--no-fit-reg", default=True, help="Estimate and plot a regression model.")
@click.option("--order", type=int, default=1, help="Order of the polynomial regression to estimate.")
@click.option("--logistic/--no-logistic", default=False, help="Fit a logistic regression model.")
def lmplot(data, output, save_data_as, x, y, hue, col, row, height, aspect, ci, scatter, fit_reg, order, logistic):
    df = load_and_handle_data(data, save_data_as)
    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "col": col,
        "row": row,
        "height": height,
        "aspect": aspect,
        "ci": ci,
        "scatter": scatter,
        "fit_reg": fit_reg,
        "order": order,
        "logistic": logistic,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.lmplot(data=df, {param_str})")

    g = sns.lmplot(data=df, **plot_params)
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.scatterplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as")
@click.option("--x", required=True, help="Column name for the X-axis (Required).")
@click.option("--y", required=True, help="Column name for the Y-axis (Required).")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--style", help="Column name for style grouping.")
@click.option("--size", help="Column name for size grouping.")
@click.option("--palette", help="Color palette to use.")
@click.option("--hue-order", help="Order for the levels of the hue variable. Comma-separated.")
@click.option("--hue-norm", help="Normalization in data units for the hue variable.")
@click.option("--sizes", help="Min, max, or list of sizes for the size variable. Comma-separated.")
@click.option("--size-order", help="Order for the levels of the size variable. Comma-separated.")
@click.option("--size-norm", help="Normalization in data units for the size variable.")
@click.option("--markers/--no-markers", default=True, help="Show markers for the plot.")
@click.option("--style-order", help="Order for the levels of the style variable. Comma-separated.")
@click.option("--alpha", type=float, help="Proportion of the original saturation to draw colors.")
@click.option(
    "--legend", type=click.Choice(["auto", "brief", "full", "False"]), default="auto", help="How to draw the legend."
)
@click.option("--ax", help="Pre-existing axes for the plot.")
def scatterplot(
    data,
    output,
    save_data_as,
    x,
    y,
    hue,
    style,
    size,
    palette,
    hue_order,
    hue_norm,
    sizes,
    size_order,
    size_norm,
    markers,
    style_order,
    alpha,
    legend,
    ax,
):
    df = load_and_handle_data(data, save_data_as)

    if hue_order:
        hue_order = hue_order.split(",")
    if sizes:
        sizes = [float(s) for s in sizes.split(",")]
    if size_order:
        size_order = size_order.split(",")
    if style_order:
        style_order = style_order.split(",")

    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "style": style,
        "size": size,
        "palette": palette,
        "hue_order": hue_order,
        "hue_norm": hue_norm,
        "sizes": sizes,
        "size_order": size_order,
        "size_norm": size_norm,
        "markers": markers,
        "style_order": style_order,
        "alpha": alpha,
        "legend": legend,
        "ax": ax,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.scatterplot(data=df, {param_str})")

    plt.figure()
    sns.scatterplot(data=df, **plot_params)
    plt.title(f"Scatter Plot: {y} vs {x}")
    plt.tight_layout()
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.stripplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--x", required=True, help="Column name for the X-axis (Required).")
@click.option("--y", required=False, help="Column name for the Y-axis (Required).")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--jitter", type=float, help="Amount of jitter (only along the categorical axis) to apply.")
@click.option("--dodge/--no-dodge", default=False, help="Separate out the strips for different hue levels.")
@click.option("--orient", type=click.Choice(["v", "h"]), help="Orientation of the plot (vertical or horizontal).")
def stripplot(data, output, save_data_as, x, y, hue, jitter, dodge, orient):
    df = load_and_handle_data(data, save_data_as)
    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "jitter": jitter,
        "dodge": dodge,
        "orient": orient,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.stripplot(data=df, {param_str})")

    plt.figure()
    sns.stripplot(data=df, **plot_params)
    plt.title(f"Strip Plot: {y} vs {x}")
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.swarmplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--x", required=True, help="Column name for the X-axis (Required).")
@click.option("--y", required=False, help="Column name for the Y-axis (Required).")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--dodge/--no-dodge", default=False, help="Separate out the strips for different hue levels.")
@click.option("--orient", type=click.Choice(["v", "h"]), help="Orientation of the plot (vertical or horizontal).")
def swarmplot(data, output, save_data_as, x, y, hue, dodge, orient):
    df = load_and_handle_data(data, save_data_as)
    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "dodge": dodge,
        "orient": orient,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.swarmplot(data=df, {param_str})")

    plt.figure()
    sns.swarmplot(data=df, **plot_params)
    plt.title(f"Swarm Plot: {y} vs {x}")
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.catplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save dataset as.")
@click.option("--x", required=True, help="Column name for the X-axis (Required).")
@click.option("--y", required=False, help="Column name for the Y-axis (Required).")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--col", help="Column name to facet the plot across columns.")
@click.option("--row", help="Column name to facet the plot across rows.")
@click.option("--kind", type=click.Choice(["strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"]))
@click.option("--palette", help="Color palette to use.")
@click.option("--order", help="Order to plot the categorical levels in. Comma-separated")
@click.option("--hue-order", help="Order for the levels of the hue variable. Comma-separated.")
@click.option("--orient", type=click.Choice(["v", "h"]), help="Orientation of the plot (vertical or horizontal).")
@click.option("--height", type=float, default=5, help="Height (in inches) of each facet.")
@click.option("--aspect", type=float, default=1, help="Aspect ratio of each facet, so aspect * height gives the width.")
@click.option("--col-wrap", type=int, help="Wrap the column variable at this width.")
def catplot(
    data,
    output,
    save_data_as,
    x,
    y,
    hue,
    col,
    row,
    kind,
    palette,
    order,
    hue_order,
    orient,
    height,
    aspect,
    col_wrap,
):
    df = load_and_handle_data(data, save_data_as)
    if order:
        order = order.split(",")
    if hue_order:
        hue_order = hue_order.split(",")

    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "col": col,
        "row": row,
        "kind": kind,
        "palette": palette,
        "order": order,
        "hue_order": hue_order,
        "orient": orient,
        "height": height,
        "aspect": aspect,
        "col_wrap": col_wrap,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.catplot(data=df, {param_str})")

    g = sns.catplot(data=df, **plot_params)
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.pairplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--kind", type=click.Choice(["scatter", "kde", "hist", "reg"]), default="scatter")
@click.option("--diag_kind", type=click.Choice(["auto", "hist", "kde", "scatter", "reg"]))
@click.option("--palette", help="Color palette to use.")
@click.option("--height", type=float, default=2.5, help="Height (in inches) of each facet.")
@click.option("--aspect", type=float, default=1, help="Aspect ratio of each facet, so aspect * height gives the width.")
def pairplot(data, output, save_data_as, hue, kind, diag_kind, palette, height, aspect):
    df = load_and_handle_data(data, save_data_as)
    plot_params = {
        "hue": hue,
        "kind": kind,
        "diag_kind": diag_kind,
        "palette": palette,
        "height": height,
        "aspect": aspect,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.pairplot(data=df, {param_str})")

    g = sns.pairplot(data=df, **plot_params)
    g.fig.suptitle("Pair Plot")
    g.fig.tight_layout()
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.histplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--x", help="Column name for the X-axis.")
@click.option("--y", help="Column name for the Y-axis.")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--weights", help="If provided, estimate density in terms of the sum of the weights.")
@click.option("--stat", type=click.Choice(["count", "frequency", "density", "probability", "percent"]), default="count")
@click.option("--bins", type=int, help="Number of bins.")
@click.option("--binwidth", type=float, help="Width of each bin.")
@click.option("--binrange", type=float, nargs=2, help="Lower and upper bounds of the bins.")
@click.option("--discrete/--no-discrete", default=False, help="Whether the variable is discrete.")
@click.option("--cumulative/--no-cumulative", default=False, help="If True, plot the cumulative distribution.")
@click.option("--common-bins/--no-common-bins", default=True, help="If True, use the same bins for all subsets.")
@click.option("--common-norm/--no-common-norm", default=True, help="If True, normalize across all subsets.")
@click.option(
    "--common-norm/--no-common-norm",
    default=True,
    help="If True, normalize across all subsets when using a normalized statistic.",
)
@click.option("--multiple", type=click.Choice(["layer", "dodge", "stack", "fill"]), default="layer")
@click.option("--element", type=click.Choice(["bars", "step", "poly"]), default="bars")
@click.option("--fill", type=bool, default=True, help="Whether to fill the bars or lines of the histogram.")
@click.option("--shrink", type=float, default=1, help="Scale the width of the bars relative to the bin width.")
@click.option("--kde/--no-kde", default=False, help="Whether to plot a kernel density estimate.")
@click.option("--kde_kws", help="Dictionary of keyword arguments for `kdeplot`.")
@click.option("--line_kws", help="Dictionary of keyword arguments for `lineplot`.")
@click.option("--log_scale", type=click.Choice([True, False, "x", "y"]), help="Set the scale of the axis to log.")
@click.option("--cbar", type=bool, default=False, help="Whether to draw a colorbar.")
@click.option("--cbar_kws", help="Dictionary of keyword arguments for `matplotlib.figure.Figure.colorbar`.")
@click.option("--palette", help="Color palette to use.")
@click.option("--hue_order", help="Order for the levels of the hue variable. Comma-separated.")
@click.option("--hue_norm", help="Normalization in data units for the hue variable.")
@click.option("--color", help="Single color for the plot elements.")
@click.option("--legend", type=bool, default=True, help="Whether to draw a legend for the semantic variables.")
@click.option("--ax", help="Pre-existing axes for the plot.")
def histplot(
    data,
    output,
    save_data_as,
    x,
    y,
    hue,
    weights,
    stat,
    bins,
    binwidth,
    binrange,
    discrete,
    cumulative,
    common_bins,
    common_norm,
    multiple,
    element,
    fill,
    shrink,
    kde,
    kde_kws,
    line_kws,
    log_scale,
    cbar,
    cbar_kws,
    palette,
    hue_order,
    hue_norm,
    color,
    legend,
    ax,
):
    df = load_and_handle_data(data, save_data_as)

    # Convert comma-separated strings to lists if necessary
    if hue_order:
        hue_order = hue_order.split(",")
    if binrange:
        binrange = tuple(binrange)

    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "weights": weights,
        "stat": stat,
        "bins": bins,
        "binwidth": binwidth,
        "binrange": binrange,
        "discrete": discrete,
        "cumulative": cumulative,
        "common_bins": common_bins,
        "common_norm": common_norm,
        "multiple": multiple,
        "element": element,
        "fill": fill,
        "shrink": shrink,
        "kde": kde,
        "kde_kws": kde_kws,
        "line_kws": line_kws,
        "log_scale": log_scale,
        "cbar": cbar,
        "cbar_kws": cbar_kws,
        "palette": palette,
        "hue_order": hue_order,
        "hue_norm": hue_norm,
        "color": color,
        "legend": legend,
        "ax": ax,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.histplot(data=df, {param_str})")

    plt.figure()
    sns.histplot(data=df, **plot_params)
    plt.title(f"Histogram: {x}")
    plt.tight_layout()
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.displot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--x", help="Column name for the X-axis.")
@click.option("--y", help="Column name for the Y-axis.")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--row", help="Column name to facet the plot across rows.")
@click.option("--col", help="Column name to facet the plot across columns.")
@click.option("--kind", type=click.Choice(["hist", "kde", "ecdf"]), default="hist")
@click.option("--height", type=float, default=5, help="Height (in inches) of each facet.")
@click.option("--aspect", type=float, default=1, help="Aspect ratio of each facet, so aspect * height gives the width.")
@click.option("--col-wrap", type=int, help="Wrap the column variable at this width.")
@click.option("--row-order", help="Order for the levels of the row variable. Comma-separated.")
@click.option("--col-order", help="Order for the levels of the col variable. Comma-separated.")
@click.option("--palette", help="Color palette to use.")
@click.option("--hue-order", help="Order for the levels of the hue variable. Comma-separated.")
@click.option("--hue-norm", help="Normalization in data units for the hue variable.")
@click.option("--color", help="Single color for the plot elements.")
@click.option("--legend/--no-legend", default=True, help="Whether to draw a legend for the semantic variables.")
@click.option("--bins", type=int, help="Number of bins.")
@click.option("--binwidth", type=float, help="Width of each bin.")
@click.option("--binrange", type=float, nargs=2, help="Lower and upper bounds of the bins.")
@click.option("--log_scale", type=click.Choice([True, False, "x", "y"]), help="Set the scale of the axis to log.")
def displot(
    data,
    output,
    save_data_as,
    x,
    y,
    hue,
    row,
    col,
    kind,
    height,
    aspect,
    col_wrap,
    row_order,
    col_order,
    palette,
    hue_order,
    hue_norm,
    color,
    legend,
    bins,
    binwidth,
    binrange,
    log_scale,
):
    df = load_and_handle_data(data, save_data_as)
    # Convert comma-separated strings to lists if necessary
    if row_order:
        row_order = row_order.split(",")
    if col_order:
        col_order = col_order.split(",")
    if hue_order:
        hue_order = hue_order.split(",")
    if binrange:
        binrange = tuple(binrange)

    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "row": row,
        "col": col,
        "kind": kind,
        "height": height,
        "aspect": aspect,
        "col_wrap": col_wrap,
        "row_order": row_order,
        "col_order": col_order,
        "palette": palette,
        "hue_order": hue_order,
        "hue_norm": hue_norm,
        "color": color,
        "legend": legend,
        "bins": bins,
        "binwidth": binwidth,
        "binrange": binrange,
        "log_scale": log_scale,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.displot(data=df, {param_str})")

    g = sns.displot(data=df, **plot_params)
    g.fig.suptitle(f"Distribution Plot: {x}")
    g.fig.tight_layout()
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.boxplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--x", help="Column name for the X-axis.")
@click.option("--y", help="Column name for the Y-axis.")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--order", help="Order to plot the categorical levels in. Comma-separated")
@click.option("--hue-order", help="Order for the levels of the hue variable. Comma-separated.")
@click.option("--orient", type=click.Choice(["v", "h"]), help="Orientation of the plot (vertical or horizontal).")
@click.option("--color", help="Single color for the plot elements.")
@click.option("--palette", help="Color palette to use.")
@click.option("--saturation", type=float, default=1, help="Proportion of the original saturation to draw colors.")
@click.option("--fill", type=bool, default=True, help="If True, use a solid patch. Otherwise, draw as line art.")
@click.option(
    "--dodge",
    type=bool,
    default=True,
    help="When hue mapping is used, whether elements should be shifted along the orient axis to eliminate overlap.",
)
@click.option("--width", type=float, default=0.8, help="Width of a full element when not using hue nesting.")
@click.option(
    "--gap",
    type=float,
    default=0,
    help="Shrink on the orient axis by this factor to add a gap between dodged elements.",
)
@click.option("--fliersize", type=float, default=5, help="Size of the markers used to indicate outlier observations.")
@click.option("--linewidth", type=float, help="Width of the lines that frame the plot elements.")
@click.option("--linecolor", help="Color to use for line elements, when `fill` is True.")
@click.option(
    "--whis",
    type=float,
    default=1.5,
    help="Proportion of the IQR past the low and high quartiles to extend the whiskers.",
)
@click.option("--hue-norm", help="Normalization in data units for the hue variable.")
@click.option("--log-scale", type=click.Choice(["x", "y"]), help="Set axis scale(s) to log.")
@click.option(
    "--native-scale",
    type=bool,
    default=False,
    help="When True, numeric or datetime values on the categorical axis will maintain their original scaling.",
)
@click.option("--formatter", help="Function for converting categorical data into strings.")
@click.option(
    "--legend", type=click.Choice(["auto", "brief", "full", "False"]), default="auto", help="How to draw the legend."
)
@click.option("--ax", help="Pre-existing axes for the plot.")
def boxplot(
    data,
    output,
    save_data_as,
    x,
    y,
    hue,
    order,
    hue_order,
    orient,
    color,
    palette,
    saturation,
    width,
    fill,
    dodge,
    gap,
    fliersize,
    linewidth,
    linecolor,
    whis,
    hue_norm,
    log_scale,
    native_scale,
    formatter,
    legend,
    ax,
):
    df = load_and_handle_data(data, save_data_as)
    # Convert comma-separated strings to lists if necessary
    if order:
        order = order.split(",")
    if hue_order:
        hue_order = hue_order.split(",")

    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "order": order,
        "hue_order": hue_order,
        "orient": orient,
        "color": color,
        "palette": palette,
        "saturation": saturation,
        "width": width,
        "fill": fill,
        "dodge": dodge,
        "gap": gap,
        "fliersize": fliersize,
        "linewidth": linewidth,
        "linecolor": linecolor,
        "whis": whis,
        "hue_norm": hue_norm,
        "log_scale": log_scale,
        "native_scale": native_scale,
        "formatter": formatter,
        "legend": legend,
        "ax": ax,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.boxplot(data=df, {param_str})")

    plt.figure()
    sns.boxplot(data=df, **plot_params)
    plt.title(f"Box Plot: {y} by {x}")
    plt.tight_layout()
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.countplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--x", help="Column name for the X-axis.")
@click.option("--y", help="Column name for the Y-axis.")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--order", help="Order to plot the categorical levels in. Comma-separated")
@click.option("--hue-order", help="Order for the levels of the hue variable. Comma-separated.")
@click.option("--orient", type=click.Choice(["v", "h"]), help="Orientation of the plot (vertical or horizontal).")
@click.option("--color", help="Single color for the plot elements.")
@click.option("--palette", help="Color palette to use.")
@click.option("--saturation", type=float, default=1, help="Proportion of the original saturation to draw colors.")
@click.option("--fill", type=bool, default=True, help="If True, use a solid patch. Otherwise, draw as line art.")
@click.option("--hue-norm", help="Normalization in data units for the hue variable.")
@click.option(
    "--stat",
    type=click.Choice(["count", "percent", "proportion", "probability"]),
    default="count",
    help="Statistic to compute.",
)
@click.option("--width", type=float, default=0.8, help="Width of a full element when not using hue nesting.")
@click.option(
    "--dodge",
    type=click.Choice(["auto", "True", "False"]),
    default="auto",
    help="When hue mapping is used, whether elements should be shifted.",
)
@click.option("--gap", type=float, default=0, help="Shrink on the orient axis by this factor to add a gap.")
@click.option("--log_scale", type=click.Choice(["x", "y"]), help="Set axis scale(s) to log.")
@click.option(
    "--native_scale",
    type=bool,
    default=False,
    help="When True, numeric or datetime values on the categorical axis will maintain their original scaling.",
)
@click.option("--formatter", help="Function for converting categorical data into strings.")
@click.option(
    "--legend", type=click.Choice(["auto", "brief", "full", "False"]), default="auto", help="How to draw the legend."
)
@click.option("--ax", help="Pre-existing axes for the plot.")
def countplot(
    data,
    output,
    save_data_as,
    x,
    y,
    hue,
    order,
    hue_order,
    orient,
    color,
    palette,
    saturation,
    fill,
    hue_norm,
    stat,
    width,
    dodge,
    gap,
    log_scale,
    native_scale,
    formatter,
    legend,
    ax,
):
    df = load_and_handle_data(data, save_data_as)
    if dodge in ["True", "False"]:
        dodge = dodge == "True"
    # Convert comma-separated strings to lists if necessary
    if order:
        order = order.split(",")
    if hue_order:
        hue_order = hue_order.split(",")

    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "order": order,
        "hue_order": hue_order,
        "orient": orient,
        "color": color,
        "palette": palette,
        "saturation": saturation,
        "fill": fill,
        "hue_norm": hue_norm,
        "stat": stat,
        "width": width,
        "dodge": dodge,
        "gap": gap,
        "log_scale": log_scale,
        "native_scale": native_scale,
        "formatter": formatter,
        "legend": legend,
        "ax": ax,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.countplot(data=df, {param_str})")

    plt.figure()
    sns.countplot(data=df, **plot_params)
    plt.title(f"Count Plot: {y} by {x}")
    plt.tight_layout()
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.lineplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--x", required=True, help="Column name for the X-axis (Required).")
@click.option("--y", required=True, help="Column name for the Y-axis (Required).")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--size", help="Column name for size grouping.")
@click.option("--style", help="Column name for style grouping.")
@click.option("--units", help="Column name for units grouping.")
@click.option("--weights", help="Data values or column used to compute weighted estimation.")
@click.option("--palette", help="Color palette to use.")
@click.option("--hue-order", help="Order for the levels of the hue variable. Comma-separated.")
@click.option("--hue-norm", help="Normalization in data units for the hue variable.")
@click.option("--sizes", help="Min, max, or list of sizes for the size variable. Comma-separated.")
@click.option("--size-order", help="Order for the levels of the size variable. Comma-separated.")
@click.option("--size-norm", help="Normalization in data units for the size variable.")
@click.option("--dashes", help="True, False, or list of dash styles for the style variable. Comma-separated.")
@click.option("--markers", type=bool, default=False, help="Show markers for the plot.")
@click.option("--style-order", help="Order for the levels of the style variable. Comma-separated.")
@click.option(
    "--estimator",
    type=click.Choice(["mean", "sum", "median"]),
    default="mean",
    help="Method for aggregating across multiple observations.",
)
@click.option(
    "--errorbar",
    help="Name of errorbar method (ci, pi, se, or sd) or a tuple with a method name and a level parameter.",
)
@click.option("--n_boot", type=int, default=1000, help="Number of bootstrap samples to use.")
@click.option("--seed", type=int, help="Seed for the random number generator.")
@click.option(
    "--orient",
    type=click.Choice(["x", "y"]),
    default="x",
    help="Dimension along which the data are sorted / aggregated.",
)
@click.option("--sort", type=bool, default=True, help="If True, the data will be sorted by the x and y variables.")
@click.option("--ci/--no-ci", default=True, help="Whether to draw confidence intervals.")
@click.option("--err_style", type=click.Choice(["band", "bars"]), default="band", help="Style of error representation.")
@click.option("--err_kws", help="Additional parameters to control the aesthetics of the error bars.")
@click.option("--alpha", type=float, help="Proportion of the original saturation to draw colors.")
@click.option("--color", help="Single color for the plot elements.")
@click.option("--linewidth", type=float, help="Width of the lines that frame the plot elements.")
@click.option(
    "--legend", type=click.Choice(["auto", "brief", "full", "False"]), default="auto", help="How to draw the legend."
)
@click.option("--ax", help="Pre-existing axes for the plot.")
@click.option("--ci", type=click.Choice([True, False]), default=True)
def lineplot(
    data,
    output,
    save_data_as,
    x,
    y,
    hue,
    size,
    style,
    units,
    weights,
    palette,
    hue_order,
    hue_norm,
    sizes,
    size_order,
    size_norm,
    dashes,
    markers,
    style_order,
    estimator,
    errorbar,
    n_boot,
    seed,
    orient,
    sort,
    ci,
    err_style,
    err_kws,
    alpha,
    color,
    linewidth,
    legend,
    ax,
):
    df = load_and_handle_data(data, save_data_as)

    # Convert comma-separated strings to lists if necessary
    if hue_order:
        hue_order = hue_order.split(",")
    if sizes:
        sizes = [float(s) for s in sizes.split(",")]
    if size_order:
        size_order = size_order.split(",")
    if dashes:
        if dashes == "True":
            dashes = True
        elif dashes == "False":
            dashes = False
        else:
            dashes = dashes.split(",")
    if style_order:
        style_order = style_order.split(",")

    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "size": size,
        "style": style,
        "units": units,
        "weights": weights,
        "palette": palette,
        "hue_order": hue_order,
        "hue_norm": hue_norm,
        "sizes": sizes,
        "size_order": size_order,
        "size_norm": size_norm,
        "dashes": dashes,
        "markers": markers,
        "style_order": style_order,
        "estimator": estimator,
        "errorbar": errorbar,
        "n_boot": n_boot,
        "seed": seed,
        "orient": orient,
        "sort": sort,
        "ci": ci,
        "err_style": err_style,
        "err_kws": err_kws,
        "alpha": alpha,
        "color": color,
        "linewidth": linewidth,
        "legend": legend,
        "ax": ax,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.lineplot(data=df, {param_str})")

    plt.figure()
    sns.lineplot(data=df, **plot_params)
    plt.title(f"Line Plot: {y} vs {x}")
    plt.tight_layout()
    save_or_show_plot(output)


# https://seaborn.pydata.org/generated/seaborn.relplot.html
@cli.command()
@click.option("--data", "-d", required=True, help="Path to data file or name of a built-in Seaborn dataset.")
@click.option("--output", "-o", type=click.Path(), help="Path to output PNG. If omitted, the plot is on screen.")
@click.option("--save-data-as", "-s", type=click.Path(), help="Save data as.")
@click.option("--x", required=True, help="Column name for the X-axis (Required).")
@click.option("--y", required=True, help="Column name for the Y-axis (Required).")
@click.option("--hue", help="Column name for color grouping.")
@click.option("--size", help="Column name for size grouping.")
@click.option("--style", help="Column name for style grouping.")
@click.option("--row", help="Column name to facet the plot across rows.")
@click.option("--col", help="Column name to facet the plot across columns.")
@click.option("--kind", type=click.Choice(["scatter", "line"]), default="scatter")
@click.option("--height", type=float, default=5, help="Height (in inches) of each facet.")
@click.option("--aspect", type=float, default=1, help="Aspect ratio of each facet, so aspect * height gives the width.")
@click.option("--col_wrap", type=int, help="Wrap the column variable at this width.")
@click.option("--row-order", help="Order for the levels of the row variable. Comma-separated.")
@click.option("--col-order", help="Order for the levels of the col variable. Comma-separated.")
@click.option("--palette", help="Color palette to use.")
@click.option("--hue-order", help="Order for the levels of the hue variable. Comma-separated.")
@click.option("--hue-norm", help="Normalization in data units for the hue variable.")
@click.option("--sizes", help="Min, max, or list of sizes for the size variable. Comma-separated.")
@click.option("--size-order", help="Order for the levels of the size variable. Comma-separated.")
@click.option("--size-norm", help="Normalization in data units for the size variable.")
@click.option("--markers", type=bool, default=True, help="Show markers for the plot.")
@click.option("--dashes", help="True, False, or list of dash styles for the style variable. Comma-separated.")
@click.option("--style-order", help="Order for the levels of the style variable. Comma-separated.")
@click.option(
    "--legend", type=click.Choice(["auto", "brief", "full", "False"]), default="auto", help="How to draw the legend."
)
@click.option("--alpha", type=float, help="Proportion of the original saturation to draw colors.")
@click.option("--color", help="Single color for the plot elements.")
@click.option("--linewidth", type=float, help="Width of the lines that frame the plot elements.")
def relplot(
    data,
    output,
    save_data_as,
    x,
    y,
    hue,
    size,
    style,
    row,
    col,
    kind,
    height,
    aspect,
    col_wrap,
    row_order,
    col_order,
    palette,
    hue_order,
    hue_norm,
    sizes,
    size_order,
    size_norm,
    markers,
    dashes,
    style_order,
    legend,
    alpha,
    color,
    linewidth,
):
    df = load_and_handle_data(data, save_data_as)

    # Convert comma-separated strings to lists if necessary
    if row_order:
        row_order = row_order.split(",")
    if col_order:
        col_order = col_order.split(",")
    if hue_order:
        hue_order = hue_order.split(",")
    if sizes:
        sizes = [float(s) for s in sizes.split(",")]
    if size_order:
        size_order = size_order.split(",")
    if dashes:
        if dashes == "True":
            dashes = True
        elif dashes == "False":
            dashes = False
        else:
            dashes = dashes.split(",")
    if style_order:
        style_order = style_order.split(",")

    plot_params = {
        "x": x,
        "y": y,
        "hue": hue,
        "size": size,
        "style": style,
        "row": row,
        "col": col,
        "kind": kind,
        "height": height,
        "aspect": aspect,
        "col_wrap": col_wrap,
        "row_order": row_order,
        "col_order": col_order,
        "palette": palette,
        "hue_order": hue_order,
        "hue_norm": hue_norm,
        "sizes": sizes,
        "size_order": size_order,
        "size_norm": size_norm,
        "markers": markers,
        "dashes": dashes,
        "style_order": style_order,
        "legend": legend,
        "alpha": alpha,
        "color": color,
        "linewidth": linewidth,
    }
    plot_params = {k: v for k, v in plot_params.items() if v is not None}
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in plot_params.items()])
    click.echo(f"sns.relplot(data=df, {param_str})")

    g = sns.relplot(data=df, **plot_params)
    g.fig.suptitle(f"Relational Plot: {y} vs {x}")
    g.fig.tight_layout()
    save_or_show_plot(output)


if __name__ == "__main__":
    # If no arguments are provided, show the help message
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    cli()
