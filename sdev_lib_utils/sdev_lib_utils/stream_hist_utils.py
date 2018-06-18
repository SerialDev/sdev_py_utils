""" Utilities to use with streaming histograms"""


import numpy as np
import matplotlib.pyplot as plt


def get_sum(stream_hist, linspace=1000):
    l, u = stream_hist.bounds()
    x = np.linspace(l, u, linspace)

    sum = [stream_hist.sum(z) for z in x]

    return sum


def get_density(stream_hist, linspace=1000):
    l, u = stream_hist.bounds()
    x = np.linspace(l, u, linspace)

    density = [stream_hist.density(z) for z in x]

    return density


def get_cdf(stream_hist, linspace=1000):
    l, u = stream_hist.bounds()
    x = np.linspace(l, u, linspace)

    cumulative_dist_func = [stream_hist.cdf(z) for z in x]

    return cumulative_dist_func


def get_pdf(stream_hist, linspace=1000):
    l, u = stream_hist.bounds()
    x = np.linspace(l, u, linspace)

    probability_density_func = [stream_hist.pdf(z) for z in x]

    return probability_density_func



# -----------------{Viz}----------------#

# TODO make an expandable interface
def save_figure(con, data, game_id, info, fig_type):
    try:
        _save_figure(con, data, game_id, info, fig_type, "figs")
    except Exception:
        update_figure(con, data, game_id, info, fig_type, "figs")


def plot_quantiles(dist_data, game_id, info, resolution, con):
    quantiles = dist_data.quantiles(0.0, 0.05, 0.1, 0.15, 0.2,
                                    0.25, 0.3, 0.35, 0.4, 0.45,
                                    0.5, 0.55, 0.6, 0.65, 0.7,
                                    0.75, 0.8, 0.85, 0.9, 0.95,
                                    1)
    x = [i/20 for i in range(0, 21, 1)]
    plt.plot(x, quantiles)
    if resolution == 'tail':
        plt.gca().set_xscale('probability', points=x, vmin=0.1)
    elif resolution == 'head':
        plt.gca().set_xscale('prob_scale', upper= 1, lower=0.01)
    else:
        pass
    plt.grid(True)
    plt.title(info)
    plt.legend(('quantiles',), loc='upper right')
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figfile_png = base64.b64encode(figfile.getvalue())
    save_figure(con, figfile_png, game_id, info, "_quantiles")
    return figfile


def plot_cdf_pdf(dist_data, game_id, info, linspace, con):
    cumulative_distribution_function = get_cdf(dist_data, linspace)
    probability_density_function = get_pdf(dist_data, linspace)
    plt.plot(cumulative_distribution_function)
    plt.plot(probability_density_function)
    plt.title(info)
    plt.legend(('cumulative dist func', 'prob dens func'), loc='upper right')

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figfile_png = base64.b64encode(figfile.getvalue())
    save_figure(con, figfile_png, game_id, info, "_cdf")

    return True


def plot_sum_and_density(dist_data, game_id, info, linspace, con):
    sum_at_point = get_sum(dist_data, linspace)
    density_at_point = get_density(dist_data, linspace)
    plt.plot(sum_at_point)
    plt.plot(density_at_point)
    plt.title(info)
    plt.legend(('Sum at point', 'Density at point'), loc='upper right')

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figfile_png = base64.b64encode(figfile.getvalue())
    save_figure(con, figfile_png, game_id, info, "_s_d")

    return True

        
