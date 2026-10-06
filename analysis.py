import csv
import os

from statistics import (
    mean,
    median,
    stdev
)

import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu


REF = (1.1, 1.1)


def non_dom_pts(pts):
    res = []

    for p in pts:
        dom = False

        for q in pts:

            if p == q:
                continue

            no_worse = (
                q[0] <= p[0]
                and
                q[1] <= p[1]
            )

            better = (
                q[0] < p[0]
                or
                q[1] < p[1]
            )

            if no_worse and better:
                dom = True
                break

        if not dom:
            res.append(p)

    return sorted(
        set(res)
    )


def get_bounds(recs):
    b = {}

    ins_set = set(
        r["ins"]
        for r in recs
    )

    for ins in ins_set:

        pts = []

        for r in recs:

            if r["ins"] == ins:
                pts.extend(
                    r["fr"]
                )

        f1 = [
            p[0]
            for p in pts
        ]

        f2 = [
            p[1]
            for p in pts
        ]

        b[ins] = (
            min(f1),
            max(f1),
            min(f2),
            max(f2)
        )

    return b


def norm_pt(p, b):
    min1, max1, min2, max2 = b

    if max1 == min1:
        x = 0.0
    else:
        x = (
            p[0] - min1
        ) / (
            max1 - min1
        )

    if max2 == min2:
        y = 0.0
    else:
        y = (
            p[1] - min2
        ) / (
            max2 - min2
        )

    return x, y


def hypervolume(
    pts,
    b,
    ref=REF
):
    pts = [
        norm_pt(p, b)
        for p in pts
    ]

    pts = non_dom_pts(pts)

    pts.sort(
        key=lambda p: p[0]
    )

    hv = 0.0
    cur_y = ref[1]

    for x, y in pts:

        if (
            x >= ref[0]
            or
            y >= ref[1]
        ):
            continue

        if y < cur_y:

            w = ref[0] - x
            h = cur_y - y

            hv += w * h

            cur_y = y

    return hv


def add_metrics(recs):
    b = get_bounds(recs)

    for r in recs:

        r["hv"] = hypervolume(
            r["fr"],
            b[r["ins"]]
        )

        r["nd"] = len(
            non_dom_pts(
                r["fr"]
            )
        )

    return b


def calc_stats(recs):
    groups = {}

    for r in recs:

        key = (
            r["cat"],
            r["ins"],
            r["cfg"],
            r["alg"]
        )

        groups.setdefault(
            key,
            []
        ).append(r)

    summ = []

    for key, vals in groups.items():

        cat, ins, cfg, alg = key

        hvs = [
            r["hv"]
            for r in vals
        ]

        nds = [
            r["nd"]
            for r in vals
        ]

        ts = [
            r["t"]
            for r in vals
        ]

        summ.append({
            "cat": cat,
            "ins": ins,
            "cfg": cfg,
            "alg": alg,
            "mean_hv": mean(hvs),
            "std_hv": (
                stdev(hvs)
                if len(hvs) > 1
                else 0.0
            ),
            "best_hv": max(hvs),
            "worst_hv": min(hvs),
            "mean_nd": mean(nds),
            "mean_t": mean(ts)
        })

    return summ


def stat_tests(recs):
    groups = {}

    for r in recs:

        key = (
            r["ins"],
            r["cfg"]
        )

        groups.setdefault(
            key,
            {}
        )

        groups[key].setdefault(
            r["alg"],
            []
        )

        groups[key][r["alg"]].append(
            r["hv"]
        )

    tests = []

    for key, vals in groups.items():

        if (
            "NSGA-II" not in vals
            or
            "SPEA2" not in vals
        ):
            continue

        ins, cfg = key

        test = mannwhitneyu(
            vals["NSGA-II"],
            vals["SPEA2"],
            alternative="two-sided"
        )

        tests.append({
            "ins": ins,
            "cfg": cfg,
            "test": "Mann-Whitney U",
            "stat": test.statistic,
            "p": test.pvalue
        })

    return tests


def save_runs(recs, path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "Category",
            "Instance",
            "Config",
            "Algorithm",
            "Run",
            "Seed",
            "HV",
            "Non-dominated",
            "Time"
        ])

        for r in recs:

            w.writerow([
                r["cat"],
                r["ins"],
                r["cfg"],
                r["alg"],
                r["run"],
                r["seed"],
                f"{r['hv']:.3f}",
                r["nd"],
                f"{r['t']:.3f}"
            ])


def save_fronts(recs, path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "Category",
            "Instance",
            "Config",
            "Algorithm",
            "Run",
            "f1",
            "f2"
        ])

        for r in recs:

            for f1, f2 in r["fr"]:

                w.writerow([
                    r["cat"],
                    r["ins"],
                    r["cfg"],
                    r["alg"],
                    r["run"],
                    f"{f1:.3f}",
                    f"{f2:.3f}"
                ])


def save_summary(summ, path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "Category",
            "Instance",
            "Config",
            "Algorithm",
            "Mean HV",
            "Std HV",
            "Best HV",
            "Worst HV",
            "Mean ND",
            "Mean time"
        ])

        for r in summ:

            w.writerow([
                r["cat"],
                r["ins"],
                r["cfg"],
                r["alg"],
                f"{r['mean_hv']:.3f}",
                f"{r['std_hv']:.3f}",
                f"{r['best_hv']:.3f}",
                f"{r['worst_hv']:.3f}",
                f"{r['mean_nd']:.3f}",
                f"{r['mean_t']:.3f}"
            ])


def save_tests(tests, path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "Instance",
            "Config",
            "Test",
            "Statistic",
            "P-value"
        ])

        for r in tests:

            w.writerow([
                r["ins"],
                r["cfg"],
                r["test"],
                f"{r['stat']:.3f}",
                f"{r['p']:.3f}"
            ])


def med_run(recs):
    hv_med = median(
        r["hv"]
        for r in recs
    )

    return min(
        recs,
        key=lambda r:
        abs(r["hv"] - hv_med)
    )

# Plot the pareto front
def plot_pareto(
    recs,
    ins,
    cfg,
    path
):
    vals = [
        r
        for r in recs
        if (
            r["ins"] == ins
            and
            r["cfg"] == cfg
        )
    ]

    plt.figure(
        figsize=(8, 6)
    )

    for alg in [
        "NSGA-II",
        "SPEA2"
    ]:

        alg_recs = [
            r
            for r in vals
            if r["alg"] == alg
        ]

        if not alg_recs:
            continue

        r = med_run(
            alg_recs
        )

        fr = non_dom_pts(
            r["fr"]
        )

        x = [
            p[0]
            for p in fr
        ]

        y = [
            p[1]
            for p in fr
        ]

        plt.scatter(
            x,
            y,
            label=alg
        )

    plt.xlabel(
        "Facility-opening cost (f1)"
    )

    plt.ylabel(
        "Customer-allocation cost (f2)"
    )

    plt.title(
        f"Pareto front - {ins} - {cfg}"
    )

    plt.legend()
    plt.grid()

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()