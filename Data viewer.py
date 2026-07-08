import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import NullFormatter
import DigitalTwin
import DataGeneration

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

HISTOGRAM_FONT_SIZE = 12
HISTOGRAM_TITLE_FONT_SIZE = 14

def load_results(results_name, filename=None):
    if filename is None:
        filename = f"{results_name}.pkl"
    
    with open(filename, 'rb') as f:
        results_dict = pickle.load(f)
    
    globals()[results_name] = results_dict[results_name]
    print(f"{results_name} loaded from {filename}")
    return results_dict[results_name]

def tuple_list_to_2D_list(tuples_list):
    array_2d = [list(item) for item in tuples_list]
    array_2d = np.swapaxes(array_2d,0,1)
    return array_2d

def average_over_x(tuples_list):
    y_values_by_x = {}
    for x, y in tuples_list:
        y_values_by_x.setdefault(x, []).append(y)
    average_y_by_x = [(k, sum(v)/len(v)) for k, v in sorted(y_values_by_x.items())]
    return np.swapaxes(average_y_by_x,0,1)

def candlestick_from_scatter(tuples_list):
    y_values_by_x = {}
    for x, y in tuples_list:
        y_values_by_x.setdefault(x, []).append(y)

    sorted_x = sorted(y_values_by_x)
    x = np.asarray(sorted_x, dtype=float)
    median_values = np.asarray([np.percentile(y_values_by_x[x_value], 50) for x_value in sorted_x], dtype=float)
    lower_quartile_values = np.asarray([np.percentile(y_values_by_x[x_value], 25) for x_value in sorted_x], dtype=float)
    upper_quartile_values = np.asarray([np.percentile(y_values_by_x[x_value], 75) for x_value in sorted_x], dtype=float)
    low_values = np.asarray([min(y_values_by_x[x_value]) for x_value in sorted_x], dtype=float)
    high_values = np.asarray([max(y_values_by_x[x_value]) for x_value in sorted_x], dtype=float)
    return x, median_values, lower_quartile_values, upper_quartile_values, low_values, high_values

def std_dev_candlestick_from_scatter(tuples_list):
    y_values_by_x = {}
    for x, y in tuples_list:
        y_values_by_x.setdefault(x, []).append(y)

    sorted_x = sorted(y_values_by_x)
    x = np.asarray(sorted_x, dtype=float)
    mean_values = np.asarray([np.mean(y_values_by_x[x_value]) for x_value in sorted_x], dtype=float)
    std_dev_values = np.asarray([np.std(y_values_by_x[x_value]) for x_value in sorted_x], dtype=float)
    low_values = np.asarray([min(y_values_by_x[x_value]) for x_value in sorted_x], dtype=float)
    high_values = np.asarray([max(y_values_by_x[x_value]) for x_value in sorted_x], dtype=float)
    return x, mean_values, std_dev_values, low_values, high_values

def _load_positive_xy_from_file(file_name):
    data = tuple_list_to_2D_list(load_results(file_name))
    if np.any(data[0] <= 0) or np.any(data[1] <= 0):
        raise ValueError("Logarithmic axes require all x and y values to be positive.")
    return np.asarray(data[0], dtype=float), np.asarray(data[1], dtype=float)

def _default_distribution_bins(x):
    return min(40, max(12, int(np.sqrt(len(x)))))

def _log_bin_edges(values, bins):
    return np.geomspace(np.min(values), np.max(values), bins + 1)

def _log_axis_ticks(values, count=5):
    lower = float(np.min(values))
    upper = float(np.max(values))
    if np.isclose(lower, upper):
        return np.asarray([lower], dtype=float)
    tick_count = max(count, 2)
    ticks = np.geomspace(lower, upper, tick_count)
    ticks[0] = lower
    ticks[-1] = upper
    return ticks

def _binned_mean_line(x, y, bin_edges):
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    means = []
    mean_centers = []
    for i, center in enumerate(centers):
        if i == len(centers) - 1:
            mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
        else:
            mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if np.any(mask):
            mean_centers.append(center)
            means.append(np.mean(y[mask]))
    return np.asarray(mean_centers, dtype=float), np.asarray(means, dtype=float)

def _binned_median_line(x, y, bin_edges):
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    medians = []
    median_centers = []
    for i, center in enumerate(centers):
        if i == len(centers) - 1:
            mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
        else:
            mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if np.any(mask):
            median_centers.append(center)
            medians.append(np.median(y[mask]))
    return np.asarray(median_centers, dtype=float), np.asarray(medians, dtype=float)

def _plot_binned_mean_line(ax, x, y, bins, color="#7f1d1d", linewidth=2.0):
    bin_edges = _log_bin_edges(x, bins)
    mean_x, mean_y = _binned_mean_line(x, y, bin_edges)
    ax.plot(mean_x, mean_y, color=color, linewidth=linewidth, zorder=5)

def _plot_binned_median_line(ax, x, y, bins, color="#7f1d1d", linewidth=2.0):
    bin_edges = _log_bin_edges(x, bins)
    median_x, median_y = _binned_median_line(x, y, bin_edges)
    ax.plot(median_x, median_y, color=color, linewidth=linewidth, zorder=5)

def _distribution_axes(fig, ax, xlabel, ylabel, title):
    _format_report_axes(fig, ax, xlabel, ylabel, title)
    ax.autoscale_view()

def _scientific_label(value, sig_figs=3):
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"${mantissa:.{sig_figs}g} \\times 10^{{{exponent}}}$"

def _lower_top_y_tick_label(ax, fig, offset_points=2):
    yticklabels = ax.get_yticklabels()
    if not yticklabels:
        return
    yticklabels[-1].set_transform(
        yticklabels[-1].get_transform() + ScaledTranslation(0, -offset_points / 72, fig.dpi_scale_trans)
    )

def _wrap_title_to_two_lines_if_needed(fig, ax, title, title_fontsize=None):
    if not title or "\n" in title:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_props = ax.title.get_fontproperties()
    title_width = renderer.get_text_width_height_descent(
        title,
        title_props,
        ismath=title.startswith("$") and title.endswith("$"),
    )[0]
    axes_width = ax.get_window_extent(renderer=renderer).width
    if title_width <= axes_width:
        return

    candidates = title.split()
    if len(candidates) < 2:
        midpoint = len(title) // 2
        candidates = [title[:midpoint].rstrip(), title[midpoint:].lstrip()]
    best_split = None
    best_score = None
    for split_index in range(1, len(candidates)):
        first_line = " ".join(candidates[:split_index]).strip()
        second_line = " ".join(candidates[split_index:]).strip()
        if not first_line or not second_line:
            continue
        first_width = renderer.get_text_width_height_descent(first_line, title_props, ismath=False)[0]
        second_width = renderer.get_text_width_height_descent(second_line, title_props, ismath=False)[0]
        score = max(first_width, second_width)
        if best_score is None or score < best_score:
            best_score = score
            best_split = (first_line, second_line)

    if best_split is not None:
        wrapped_title_fontsize = title_fontsize if title_fontsize is not None else ax.title.get_fontsize()
        ax.set_title(f"{best_split[0]}\n{best_split[1]}", fontsize=wrapped_title_fontsize)

def _format_histogram_ticks(fig, ax, x_ticks, y_ticks):
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([_scientific_label(tick, sig_figs=2) for tick in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([_scientific_label(tick, sig_figs=2) for tick in y_ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    _lower_top_y_tick_label(ax, fig)
    ax.tick_params(axis="both", pad=6, labelsize=HISTOGRAM_FONT_SIZE)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 14
    fig.tight_layout(pad=1.5)

def _histogram_density_plot(ax, x, y, bins):
    x_edges = _log_bin_edges(x, bins)
    y_edges = _log_bin_edges(y, bins)
    hist = ax.hist2d(
        x,
        y,
        bins=[x_edges, y_edges],
        cmap="Blues",
        norm=LogNorm(),
        cmin=1,
    )
    return hist

def _parameter_name_from_results_file(file_name):
    results_name = Path(file_name).stem
    if not results_name.endswith("_results"):
        raise ValueError("Expected a results file name ending in '_results'.")
    return results_name[:-len("_results")]

def _typical_non_dimensional_parameter_value(parameter_name):
    (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, d_i, w_i, N_o, n_particles) = DataGeneration.return_typical_dimensional_parameters()
    typical_values = DataGeneration.calculate_non_dimensional_parameters(
        DigitalTwin.Particle("Particle 1", d_p, rho_p),
        DigitalTwin.Potential_flow_around_a_cylinder(U, d_c / 2, mu, rho_f),
        DigitalTwin.Camera("Camera 1", int(np.ceil(np.sqrt(R))), int(np.ceil(np.sqrt(R))), d_c / 2, d_c / 2, w_0, w_0),
        DigitalTwin.Double_exposure_properties(t_e, t_delta, I, d_i),
        DigitalTwin.Interrogation_properties(U * 2.2, w_i, N_o),
        int(n_particles),
        gravity=0,
    )
    parameter_names = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "kappa", "lambda", "xi", "phi", "psi")
    parameter_map = dict(zip(parameter_names, typical_values))
    if parameter_name not in parameter_map:
        raise ValueError(f"Unsupported parameter name '{parameter_name}'.")
    return parameter_map[parameter_name]

def _format_histogram_colorbar(fig, ax, mappable, count_max):
    colorbar = fig.colorbar(mappable, ax=ax, label="Count")
    colorbar.ax.tick_params(labelsize=HISTOGRAM_FONT_SIZE)
    colorbar.ax.yaxis.label.set_size(HISTOGRAM_FONT_SIZE)
    vmin = 1.0
    vmax = max(count_max, 1.0)
    mid = np.sqrt(vmin * vmax)
    colorbar_ticks = np.asarray([vmin, mid, vmax], dtype=float)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels(["1", _scientific_label(mid, sig_figs=2), _scientific_label(vmax, sig_figs=2)])
    return colorbar

def _format_report_axes(fig, ax, xlabel, ylabel, title, title_fontsize=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if title_fontsize is not None:
        ax.title.set_size(title_fontsize)
    _wrap_title_to_two_lines_if_needed(fig, ax, title, title_fontsize=title_fontsize)
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.2)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

def _format_histogram_axes(fig, ax, xlabel, ylabel, title):
    _format_report_axes(fig, ax, xlabel, ylabel, title, title_fontsize=HISTOGRAM_TITLE_FONT_SIZE)
    ax.xaxis.label.set_size(HISTOGRAM_FONT_SIZE)
    ax.yaxis.label.set_size(HISTOGRAM_FONT_SIZE)

def averaged_graph_from_file(
    file_name,
    xlabel="x",
    ylabel="y",
    title="Averaged data"
):
    data = average_over_x(load_results(file_name))
    if np.any(data[0] <= 0) or np.any(data[1] <= 0):
        raise ValueError("Logarithmic axes require all x and y values to be positive.")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(data[0], data[1], color="black", linewidth=1.8)
    _format_report_axes(fig, ax, xlabel, ylabel, title)
    fig.tight_layout()
    plt.show()

def scatter_from_file(
    file_name,
    xlabel="x",
    ylabel="y",
    title="Scatter data"
):
    data = tuple_list_to_2D_list(load_results(file_name))
    if np.any(data[0] <= 0) or np.any(data[1] <= 0):
        raise ValueError("Logarithmic axes require all x and y values to be positive.")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(data[0], data[1], s=18, color="black", marker="o", linewidths=0.4)
    _format_report_axes(fig, ax, xlabel, ylabel, title)
    fig.tight_layout()
    plt.show()

def candlestick_from_file(
    file_name,
    xlabel="x",
    ylabel="y",
    title="Candlestick data"
):
    x, median_values, lower_quartile_values, upper_quartile_values, low_values, high_values = candlestick_from_scatter(load_results(file_name))
    if np.any(x <= 0) or np.any(low_values <= 0):
        raise ValueError("Logarithmic axes require all x and y values to be positive.")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.asarray(x, dtype=float)
    median_values = np.asarray(median_values, dtype=float)
    lower_quartile_values = np.asarray(lower_quartile_values, dtype=float)
    upper_quartile_values = np.asarray(upper_quartile_values, dtype=float)
    low_values = np.asarray(low_values, dtype=float)
    high_values = np.asarray(high_values, dtype=float)
    candle_widths = np.maximum(x * 0.06, np.finfo(float).eps)

    for xi, median_value, lower_quartile_value, upper_quartile_value, low_value, high_value, width in zip(
        x, median_values, lower_quartile_values, upper_quartile_values, low_values, high_values, candle_widths
    ):
        ax.vlines(xi, low_value, high_value, color="#4a4a4a", linewidth=1.0, zorder=2)
        cap_width = width * 0.45
        ax.hlines(low_value, xi - cap_width / 2, xi + cap_width / 2, color="#4a4a4a", linewidth=1.0, zorder=2)
        ax.hlines(high_value, xi - cap_width / 2, xi + cap_width / 2, color="#4a4a4a", linewidth=1.0, zorder=2)

        if upper_quartile_value == lower_quartile_value:
            ax.hlines(lower_quartile_value, xi - width / 2, xi + width / 2, color="#1f3b73", linewidth=1.2, zorder=3)
        else:
            ax.add_patch(
                Rectangle(
                    (xi - width / 2, lower_quartile_value),
                    width,
                    median_value - lower_quartile_value,
                    facecolor="#d7e3f4",
                    edgecolor="#1f3b73",
                    linewidth=1.0,
                    zorder=3,
                )
            )
            ax.add_patch(
                Rectangle(
                    (xi - width / 2, median_value),
                    width,
                    upper_quartile_value - median_value,
                    facecolor="#8fb1d9",
                    edgecolor="#1f3b73",
                    linewidth=1.0,
                    zorder=3,
                )
            )
            ax.hlines(median_value, xi - width / 2, xi + width / 2, color="#1f3b73", linewidth=0.9, zorder=4)

    _format_report_axes(fig, ax, xlabel, ylabel, title)
    ax.autoscale_view()
    fig.tight_layout()
    plt.show()

def std_dev_candlestick_from_file(
    file_name,
    xlabel="x",
    ylabel="y",
    title="Std dev candlestick data"
):
    x, mean_values, std_dev_values, low_values, high_values = std_dev_candlestick_from_scatter(load_results(file_name))
    lower_band_values = mean_values - std_dev_values
    upper_band_values = mean_values + std_dev_values
    if np.any(x <= 0) or np.any(low_values <= 0):
        raise ValueError("Logarithmic axes require all x and y values to be positive.")

    positive_floor = np.min(low_values[low_values > 0]) * 0.5

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.asarray(x, dtype=float)
    mean_values = np.asarray(mean_values, dtype=float)
    lower_band_values = np.asarray(lower_band_values, dtype=float)
    upper_band_values = np.asarray(upper_band_values, dtype=float)
    low_values = np.asarray(low_values, dtype=float)
    high_values = np.asarray(high_values, dtype=float)
    candle_widths = np.maximum(x * 0.06, np.finfo(float).eps)

    for xi, mean_value, lower_band_value, upper_band_value, low_value, high_value, width in zip(
        x, mean_values, lower_band_values, upper_band_values, low_values, high_values, candle_widths
    ):
        lower_band_value = max(lower_band_value, positive_floor)
        ax.vlines(xi, low_value, high_value, color="#4a4a4a", linewidth=1.0, zorder=2)
        cap_width = width * 0.45
        ax.hlines(low_value, xi - cap_width / 2, xi + cap_width / 2, color="#4a4a4a", linewidth=1.0, zorder=2)
        ax.hlines(high_value, xi - cap_width / 2, xi + cap_width / 2, color="#4a4a4a", linewidth=1.0, zorder=2)

        if upper_band_value == lower_band_value:
            ax.hlines(lower_band_value, xi - width / 2, xi + width / 2, color="#1f3b73", linewidth=1.2, zorder=3)
        else:
            ax.add_patch(
                Rectangle(
                    (xi - width / 2, lower_band_value),
                    width,
                    mean_value - lower_band_value,
                    facecolor="#d7e3f4",
                    edgecolor="#1f3b73",
                    linewidth=1.0,
                    zorder=3,
                )
            )
            ax.add_patch(
                Rectangle(
                    (xi - width / 2, mean_value),
                    width,
                    upper_band_value - mean_value,
                    facecolor="#8fb1d9",
                    edgecolor="#1f3b73",
                    linewidth=1.0,
                    zorder=3,
                )
            )
            ax.hlines(mean_value, xi - width / 2, xi + width / 2, color="#1f3b73", linewidth=0.9, zorder=4)

    _format_report_axes(fig, ax, xlabel, ylabel, title)
    ax.autoscale_view()
    fig.tight_layout()
    plt.show()

def histogram2d_from_file(
    file_name,
    xlabel="x",
    ylabel="$E_{RMS}/U$",
    title="2D histogram"
):
    x, y = _load_positive_xy_from_file(file_name)
    bins = _default_distribution_bins(x)
    x_ticks = _log_axis_ticks(x)
    y_ticks = _log_axis_ticks(y, count=4)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    hist = _histogram_density_plot(ax, x, y, bins)
    count_max = float(np.nanmax(hist[0]))
    hist[3].set_norm(LogNorm(vmin=1.0, vmax=max(count_max, 1.0)))
    _format_histogram_colorbar(fig, ax, hist[3], count_max)
    _plot_binned_mean_line(ax, x, y, bins, color="black", linewidth=1.0)
    _format_histogram_axes(fig, ax, xlabel, ylabel, title)
    _format_histogram_ticks(fig, ax, x_ticks, y_ticks)
    plt.show()

def histogram2d_median_from_file(
    file_name,
    xlabel="x",
    ylabel="$E_{RMS}/U$",
    title="2D histogram (median)"
):
    x, y = _load_positive_xy_from_file(file_name)
    bins = _default_distribution_bins(x)
    x_ticks = _log_axis_ticks(x)
    y_ticks = _log_axis_ticks(y, count=4)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    hist = _histogram_density_plot(ax, x, y, bins)
    count_max = float(np.nanmax(hist[0]))
    hist[3].set_norm(LogNorm(vmin=1.0, vmax=max(count_max, 1.0)))
    _format_histogram_colorbar(fig, ax, hist[3], count_max)
    _plot_binned_median_line(ax, x, y, bins, color="black", linewidth=1.0)
    parameter_name = _parameter_name_from_results_file(file_name)
    typical_value = _typical_non_dimensional_parameter_value(parameter_name)
    ax.axvline(typical_value, color="red", linestyle="--", linewidth=0.8, zorder=6)
    ax.scatter([typical_value], [0.0205], marker="x", color="red", s=80, linewidths=1, zorder=7)
    ax.scatter([typical_value], [0.0205], marker="o", color="white", s=120, linewidths=1, zorder=1)
    _format_histogram_axes(fig, ax, xlabel, ylabel, title)
    _format_histogram_ticks(fig, ax, x_ticks, y_ticks)
    plt.show()

def typical_sim_results():
    (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, d_i, w_i, N_o, n_particles) = DataGeneration.return_typical_dimensional_parameters()
    image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = DataGeneration.simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, d_i, w_i, N_o, n_particles)
    DigitalTwin.plot_vector_field(interrogation_spots, velocity_vectors, d_c, U, w_0, "Measured Velocities")
    DigitalTwin.plot_vector_field(interrogation_spots, errors, d_c, U, w_0, "Velocity Measurement Errors")
