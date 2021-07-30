from pathlib import Path
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from string import Template
import pprint

PRETTY_NAMES = {
    "model_ac_opf": "Model → ACOPF",
    "model": "Model",
    "dcac_opf": "DCOPF → ACOPF",
    "dc_opf": "DCOPF",
    "ac_opf": "ACOPF"
}

def plot_histogram(results, filter_by_solved, filter_by_methods=None):
    filter_by_methods = filter_by_methods or None
    if filter_by_methods:
        results = results[~results.opf_method.isin(filter_by_methods)]
    if filter_by_solved:
        results = results[results["solved"] == 1]

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    methods = results.opf_method.unique()

    min_val = np.min(results.time_taken)
    max_val = np.max(results.time_taken)
    n_bins = 100

    # bin_lims = np.linspace(min_val, max_val, n_bins + 1)
    bin_lims = np.logspace(np.log10(min_val), np.log10(max_val), n_bins)
    bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
    bin_widths = bin_lims[1:] - bin_lims[:-1]

    for m in sorted(methods):
        mask = results.opf_method == m
        vals = results.time_taken[mask]
        hist, _ = np.histogram(vals, bins=bin_lims)
        hist = hist / np.max(hist)
        ax.bar(bin_centers, hist, width=bin_widths, align='center', label=m,
               alpha=0.5)
    ax.legend()
    plt.show()
    return fig

def strfdelta(tdelta, fmt):
    class DeltaTemplate(Template):
        delimiter = "%"
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)

def format_time(seconds):
    time = datetime.timedelta(seconds=seconds)
    if seconds == 0:
        return "< 1s"
    if seconds < 60:
        return str(seconds) + "s"
    elif seconds < 3600:
        time_format = "%M:%S min"
    else:
        time_format = "%H:%M:%S h"
    timestring = strfdelta(time, time_format)
    return timestring

def plot_relative_time(results, filter_by_solved, filter_by_methods=None):
    filter_by_methods = filter_by_methods or None
    if filter_by_methods:
        results = results[~results.opf_method.isin(filter_by_methods)]
    if filter_by_solved:
        results = results[results["solved"] == 1]

    fig, ax = plt.subplots()
    methods = sorted(results.opf_method.unique())
    runtimes = []

    for m in methods:
        mask = results.opf_method == m
        vals = results.time_taken[mask]
        mean_val = np.mean(vals)
        runtimes.append(mean_val)

    runtimes, methods = zip(*sorted(zip(runtimes, methods), reverse=True))

    pos = np.arange(len(methods))
    rects = ax.barh(pos, runtimes,
             align='center',
             height=0.5,
             tick_label=[PRETTY_NAMES.get(m, m) for m in methods])

    rect_labels = []
    for rect in rects:
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        width = int(rect.get_width())

        rankStr = format_time(width)
        # The bars aren't wide enough to print the ranking inside
        if width < 40:
            # Shift the text to the right side of the right edge
            xloc = 5
            # Black against white background
            clr = 'black'
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            xloc = -5
            # White on magenta
            clr = 'white'
            align = 'right'

        # Center the text vertically in the bar
        yloc = rect.get_y() + rect.get_height() / 2
        label = ax.annotate(rankStr, xy=(width, yloc), xytext=(xloc, 0),
                            textcoords="offset points",
                            ha=align, va='center',
                            color=clr, weight='bold', clip_on=True)
        rect_labels.append(label)
    ax.set_xlabel('Mean Runtime (s)')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    #for tick in ax.get_yticklabels():
    #    tick.set_rotation(45)

    plt.show()
    fig.set_size_inches(23, 10.5)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + rect_labels +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(26)
    fig.savefig("./relative_runtimes.svg", bbox_inches='tight')
    fig.savefig("./relative_runtimes.pdf", bbox_inches='tight')
    fig.savefig("./relative_runtimes.png", dpi=200, bbox_inches='tight')
    return fig

def solved_in_n_seconds(n_seconds, results, filter_by_methods=None):
    filter_by_methods = filter_by_methods or None
    if filter_by_methods:
        results = results[~results.opf_method.isin(filter_by_methods)]
    results = results[results["solved"] == 1]

    methods = results.opf_method.unique()

    for m in sorted(methods):
        mask = results.opf_method == m
        vals = results.time_taken[mask]
        mask_time = vals <= n_seconds
        print(f"{m} solved {np.sum(mask_time)} in {n_seconds}")


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def statistical_summary(results, filter_by_solved):
    group_by = ["opf_method"]
    if filter_by_solved:
        results = results[results["solved"] == 1]
    grouped_results = results.groupby(group_by)
    agg_dict = {c: ["mean", "std", "median", "max", percentile(95)] for c in list(results.columns.values)
                if c not in group_by + ["scenario_id", "solved"]}
    agg_dict["solved"] = ["mean", "sum"]
    statistics_df = grouped_results.agg(agg_dict)
    # statistics_df = statistics_df.unstack(level=[1]).reorder_levels([2, 0, 1], axis=1)
    # sort whole group according to test acc
    statistics_df = statistics_df.sort_values(by=[("time_taken", "mean")], ascending=True)
    return statistics_df


def pretty_statistics(results, filter_by_solved):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                           "display.width", 400):
        pretty_statistic_string = str(statistical_summary(results, filter_by_solved))
    return pretty_statistic_string


def compute_time_improvement(results, filter_by_solved, base_method="ac_opf"):
    improvements = {}
    if filter_by_solved:
        results = results[results["solved"] == 1]
    base_df = results[results.opf_method == base_method]
    methods = results.opf_method.unique()
    for m in sorted(methods):
        if m == base_method:
            continue
        method_df = results[results.opf_method == m]
        merged_df = method_df.merge(base_df, left_on="scenario_id",
                        right_on="scenario_id")
        diff = merged_df.time_taken_y - merged_df.time_taken_x
        improvements[m] = {
            "mean improvement": np.mean(diff),
            "most improvement": np.max(diff),
            "least improvement": np.min(diff),
            "median improvement": np.median(diff),
            "relative improvement": np.mean(merged_df.time_taken_y) / np.mean(merged_df.time_taken_x)
        }
    return improvements

def create_best_ac_and_model(results):
    # creates a hypothetical case in which we can choose the best of warm-starting acopf and
    # not warmstarting it.
    acopf_results = results[results.opf_method == "ac_opf"]
    model_ac_opf_results = results[results.opf_method == "model_ac_opf"]
    dcacopf_results = results[results.opf_method == "dcac_opf"]
    for scen in acopf_results.scenario_id.unique():
        acopf = acopf_results[acopf_results.scenario_id == scen]
        modelacopf = model_ac_opf_results[model_ac_opf_results.scenario_id == scen]
        dcacopf = dcacopf_results[dcacopf_results.scenario_id == scen]
        if acopf.time_taken.iloc[0] < model_ac_opf_results.time_taken.iloc[0]:
            chosen = acopf
        else:
            chosen = modelacopf
        if dcacopf.time_taken.iloc[0] < chosen.time_taken.iloc[0]:
            chosen = dcacopf
        chosen = chosen.iloc[0].to_dict()
        chosen = pd.DataFrame.from_dict({k: [v] for k, v in chosen.items()})
        chosen["opf_method"] = ["best"]
        results = pd.concat([results, chosen], ignore_index=True, axis=0, sort=False)
    print(results)
    return results


def main(results_file, base_csv=None, take_from_base=None):
    pp = pprint.PrettyPrinter(indent=4)

    results = pd.read_csv(results_file)
    if base_csv is not None:
        added_results = pd.read_csv(base_csv)
        for method in take_from_base:
            method_results = added_results[added_results["opf_method"] == method]
            results = pd.concat([results, method_results], ignore_index=True, axis=0, sort=False)
    # results = create_best_ac_and_model(results)
    print("All")
    print(pretty_statistics(results, filter_by_solved=False))
    pp.pprint(compute_time_improvement(results, filter_by_solved=False))
    print("\nSolved")
    print(pretty_statistics(results, filter_by_solved=True))
    solved_in_n_seconds(300, results)
    solved_in_n_seconds(60, results)

    pp.pprint(compute_time_improvement(results, filter_by_solved=True))

    #fig = plot_histogram(results, filter_by_solved=False, filter_by_methods=["model", "dc_opf"])
    #plt.show()

    fig = plot_relative_time(results, filter_by_solved=False)
    # print(results)


def eval_2k(exp_path):
    subpath = Path("results/results.csv")
    results_file = exp_path / subpath
    main(results_file)


def eval_200(exp_path):
    subpath = Path("results/results.csv")
    results_file = exp_path / subpath
    main(results_file)

def eval_on_params():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--exp_path", required=True, type=Path)
    args = parser.parse_args()
    if args.dataset == "ACTIVSg200":
        eval_200(args.exp_path)
    elif args.dataset == "ACTIVSg2000":
        eval_2k(args.exp_path)

if __name__ == '__main__':
    eval_on_params()
