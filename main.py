import os
import time

from problem import load_data

from algorithms import (
    nsga2,
    spea2
)

from analysis import (
    add_metrics,
    calc_stats,
    stat_tests,
    save_runs,
    save_fronts,
    save_summary,
    save_tests,
    plot_pareto
)


INS = {
    "Small": [
        "cap61",
        "cap62"
    ],

    "Medium": [
        "cap101",
        "cap102"
    ],

    "Large": [
        "cap121",
        "cap122"
    ]
}


CFGS = {
    "P1": {
        "pop_size": 50,
        "max_evals": 6000,
        "cx_p": 0.8,
        "mut_p": 0.05
    },

    "P2": {
        "pop_size": 100,
        "max_evals": 6000,
        "cx_p": 0.9,
        "mut_p": 0.10
    },

    "P3": {
        "pop_size": 150,
        "max_evals": 6000,
        "cx_p": 0.9,
        "mut_p": 0.15
    }
}


ALGS = [
    "NSGA-II",
    "SPEA2"
]


RUNS = 10


def run_alg(
    alg,
    p,
    cfg,
    seed
):
    start = time.perf_counter()

    if alg == "NSGA-II":

        fr = nsga2(
            p=p,
            pop_size=cfg["pop_size"],
            max_evals=cfg["max_evals"],
            cx_p=cfg["cx_p"],
            mut_p=cfg["mut_p"],
            seed=seed
        )

    elif alg == "SPEA2":

        fr = spea2(
            p=p,
            pop_size=cfg["pop_size"],
            max_evals=cfg["max_evals"],
            cx_p=cfg["cx_p"],
            mut_p=cfg["mut_p"],
            seed=seed
        )

    else:
        raise ValueError(
            "Unknown algorithm."
        )

    t = (
        time.perf_counter()
        - start
    )

    return fr, t


def main():
    recs = []

    os.makedirs(
        "output",
        exist_ok=True
    )

    for cat, ins_list in INS.items():

        for ins in ins_list:

            path = os.path.join(
                "data",
                ins + ".txt"
            )

            print(
                f"\nLoading {path}"
            )

            p = load_data(
                path
            )

            print(
                f"{ins}: "
                f"{p.nf} facilities, "
                f"{p.nc} customers"
            )

            for cfg_name, cfg in CFGS.items():

                print(
                    f"\n{ins} - {cfg_name}"
                )

                for run in range(
                    1,
                    RUNS + 1
                ):

                    seed = (
                        1000 + run
                    )

                    for alg in ALGS:

                        print(
                            f"  {alg} "
                            f"run {run}/{RUNS}"
                        )

                        fr, t = run_alg(
                            alg,
                            p,
                            cfg,
                            seed
                        )

                        pts = [
                            ind.obj
                            for ind in fr
                        ]

                        if (
                            ins == "cap61"
                            and cfg_name == "P2"
                            and run == 1
                            and alg == "NSGA-II"
                        ):
                            ex = fr[0]

                            print(
                                "\nExample encoded individual:"
                            )

                            print(
                                "Chromosome:",
                                ex.ch
                            )

                            print(
                                "Objectives:",
                                ex.obj
                            )

                            with open(
                                "output/example_chromosome.txt",
                                "w"
                            ) as f:

                                f.write(
                                    f"Instance: {ins}\n"
                                )

                                f.write(
                                    f"Config: {cfg_name}\n"
                                )

                                f.write(
                                    f"Algorithm: {alg}\n"
                                )

                                f.write(
                                    f"Run: {run}\n"
                                )

                                f.write(
                                    f"Seed: {seed}\n"
                                )

                                f.write(
                                    f"Chromosome: {ex.ch}\n"
                                )

                                f.write(
                                    f"f1: {ex.obj[0]:.3f}\n"
                                )

                                f.write(
                                    f"f2: {ex.obj[1]:.3f}\n"
                                )

                        recs.append({
                            "cat": cat,
                            "ins": ins,
                            "cfg": cfg_name,
                            "alg": alg,
                            "run": run,
                            "seed": seed,
                            "fr": pts,
                            "t": t
                        })

    add_metrics(
        recs
    )

    summ = calc_stats(
        recs
    )

    tests = stat_tests(
        recs
    )

    save_runs(
        recs,
        "output/run_results.csv"
    )

    save_fronts(
        recs,
        "output/fronts.csv"
    )

    save_summary(
        summ,
        "output/summary.csv"
    )

    save_tests(
        tests,
        "output/statistical_tests.csv"
    )

    plot_set = [
        "cap61",
        "cap101",
        "cap121"
    ]

    for ins in plot_set:

        plot_pareto(
            recs=recs,
            ins=ins,
            cfg="P2",
            path=os.path.join(
                "output",
                f"{ins}_pareto.png"
            )
        )

    print(
        "\nFinished."
    )

    print(
        "Results saved in output/"
    )


if __name__ == "__main__":
    main()