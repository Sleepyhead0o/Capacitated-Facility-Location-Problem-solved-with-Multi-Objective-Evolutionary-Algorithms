import math
import random
from dataclasses import dataclass

from problem import (
    make_ch,
    repair,
    eval_obj
)


@dataclass
class Ind:
    ch: list
    obj: tuple = None
    rank: int = 0
    crowd: float = 0.0
    fit: float = 0.0


def eval_ind(ind, p):
    ind.obj = eval_obj(
        ind.ch,
        p
    )

    return ind


def make_pop(p, n, rng):
    pop = []

    for _ in range(n):
        ch = make_ch(p, rng) # makes the chromomes 

        ind = Ind(ch)

        eval_ind(ind, p)

        pop.append(ind)

    return pop


def dominates(a, b):
    no_worse = (
        a.obj[0] <= b.obj[0]
        and
        a.obj[1] <= b.obj[1]
    )

    better = (
        a.obj[0] < b.obj[0]
        or
        a.obj[1] < b.obj[1]
    )

    return no_worse and better


def crossover(a, b, cx_p, p, rng):
    ch1 = a.ch.copy()
    ch2 = b.ch.copy()

    if rng.random() < cx_p:
        pt = rng.randint(
            1,
            p.nc - 1
        )

        ch1 = (
            a.ch[:pt]
            + b.ch[pt:]
        )

        ch2 = (
            b.ch[:pt]
            + a.ch[pt:]
        )

    ch1 = repair(
        ch1,
        p,
        rng
    )

    ch2 = repair(
        ch2,
        p,
        rng
    )

    return ch1, ch2


def mutate(ch, mut_p, p, rng):
    ch = ch.copy()

    if rng.random() < mut_p:
        c = rng.randrange(p.nc)

        old_f = ch[c]

        opts = [
            f
            for f in range(p.nf)
            if f != old_f
        ]

        if opts:
            ch[c] = rng.choice(opts)

    return repair(
        ch,
        p,
        rng
    )


def unique_front(pop):
    fr = []

    for a in pop:
        dom = False

        for b in pop:
            if a is b:
                continue

            if dominates(b, a):
                dom = True
                break

        if not dom:
            fr.append(a)

    uniq = {}

    for ind in fr:
        if ind.obj not in uniq:
            uniq[ind.obj] = ind

    return list(uniq.values())


# ----- NSGA-II ------

def non_dominated_sort(pop):
    n = len(pop)

    cnt = [0] * n
    dom = [[] for _ in range(n)]

    frs = [[]]

    for i in range(n):

        for j in range(n):

            if i == j:
                continue

            if dominates(
                pop[i],
                pop[j]
            ):
                dom[i].append(j)

            elif dominates(
                pop[j],
                pop[i]
            ):
                cnt[i] += 1

        if cnt[i] == 0:
            pop[i].rank = 0
            frs[0].append(i)

    k = 0

    while k < len(frs) and frs[k]:
        nxt = []

        for i in frs[k]:

            for j in dom[i]:
                cnt[j] -= 1

                if cnt[j] == 0:
                    pop[j].rank = k + 1
                    nxt.append(j)

        if nxt:
            frs.append(nxt)

        k += 1

    return [
        [pop[i] for i in fr]
        for fr in frs
        if fr
    ]


def crowding_distance(fr):
    if not fr:
        return

    for ind in fr:
        ind.crowd = 0.0

    if len(fr) <= 2:
        for ind in fr:
            ind.crowd = float("inf")

        return

    for m in range(2):

        fr.sort(
            key=lambda x: x.obj[m]
        )

        fr[0].crowd = float("inf")
        fr[-1].crowd = float("inf")

        lo = fr[0].obj[m]
        hi = fr[-1].obj[m]

        if hi == lo:
            continue

        for i in range(
            1,
            len(fr) - 1
        ):
            prev_v = fr[i - 1].obj[m]
            next_v = fr[i + 1].obj[m]

            fr[i].crowd += (
                next_v - prev_v
            ) / (
                hi - lo
            )


def prep_nsga(pop):
    frs = non_dominated_sort(pop)

    for fr in frs:
        crowding_distance(fr)

    return frs


def nsga_tour(pop, rng):
    a, b = rng.sample(
        pop,
        2
    )

    if a.rank < b.rank:
        return a

    if b.rank < a.rank:
        return b

    if a.crowd > b.crowd:
        return a

    if b.crowd > a.crowd:
        return b

    return rng.choice(
        [a, b]
    )


def nsga2(
    p,
    pop_size,
    max_evals,
    cx_p,
    mut_p,
    seed
):
    if max_evals < pop_size:
        raise ValueError(
            "max_evals must be at least pop_size."
        )

    rng = random.Random(seed)

    pop = make_pop(
        p,
        pop_size,
        rng
    )

    evals = pop_size

    while evals < max_evals:

        prep_nsga(pop)

        off = []

        n_off = min(
            pop_size,
            max_evals - evals
        )

        while len(off) < n_off:

            p1 = nsga_tour(
                pop,
                rng
            )

            p2 = nsga_tour(
                pop,
                rng
            )

            ch1, ch2 = crossover(
                p1,
                p2,
                cx_p,
                p,
                rng
            )

            ch1 = mutate(
                ch1,
                mut_p,
                p,
                rng
            )

            ch2 = mutate(
                ch2,
                mut_p,
                p,
                rng
            )

            i1 = Ind(ch1)
            eval_ind(i1, p)

            off.append(i1)
            evals += 1

            if (
                len(off) < n_off
                and
                evals < max_evals
            ):
                i2 = Ind(ch2)
                eval_ind(i2, p)

                off.append(i2)
                evals += 1

        comb = pop + off

        frs = non_dominated_sort(
            comb
        )

        new_pop = []

        for fr in frs:
            crowding_distance(fr)

            if (
                len(new_pop) + len(fr)
                <= pop_size
            ):
                new_pop.extend(fr)

            else:
                fr.sort(
                    key=lambda x: x.crowd,
                    reverse=True
                )

                n = (
                    pop_size
                    - len(new_pop)
                )

                new_pop.extend(
                    fr[:n]
                )

                break

        pop = new_pop

    return unique_front(pop)


# ----- SPEA2 ------

def norm_obj(pop):
    f1 = [
        ind.obj[0]
        for ind in pop
    ]

    f2 = [
        ind.obj[1]
        for ind in pop
    ]

    min1 = min(f1)
    max1 = max(f1)

    min2 = min(f2)
    max2 = max(f2)

    vals = {}

    for ind in pop:

        if max1 == min1:
            x = 0.0
        else:
            x = (
                ind.obj[0] - min1
            ) / (
                max1 - min1
            )

        if max2 == min2:
            y = 0.0
        else:
            y = (
                ind.obj[1] - min2
            ) / (
                max2 - min2
            )

        vals[id(ind)] = (
            x,
            y
        )

    return vals


def spea2_fitness(pop):
    n = len(pop)

    s = [0] * n
    raw = [0] * n

    for i in range(n):

        for j in range(n):

            if i == j:
                continue

            if dominates(
                pop[i],
                pop[j]
            ):
                s[i] += 1

    for i in range(n):

        for j in range(n):

            if i == j:
                continue

            if dominates(
                pop[j],
                pop[i]
            ):
                raw[i] += s[j]

    vals = norm_obj(pop)

    k = max(
        1,
        int(math.sqrt(n))
    )

    for i, ind in enumerate(pop):

        x1, y1 = vals[id(ind)]

        ds = []

        for j, other in enumerate(pop):

            if i == j:
                continue

            x2, y2 = vals[id(other)]

            d = math.sqrt(
                (x1 - x2) ** 2
                +
                (y1 - y2) ** 2
            )

            ds.append(d)

        ds.sort()

        if ds:
            pos = min(
                k - 1,
                len(ds) - 1
            )

            sig = ds[pos]

        else:
            sig = 0.0

        den = 1.0 / (
            sig + 2.0
        )

        ind.fit = (
            raw[i] + den
        )


def truncate(arc, size):
    arc = arc.copy()

    while len(arc) > size:

        vals = norm_obj(arc)

        all_ds = []

        for ind in arc:

            x1, y1 = vals[id(ind)]

            ds = []

            for other in arc:

                if ind is other:
                    continue

                x2, y2 = vals[id(other)]

                d = math.sqrt(
                    (x1 - x2) ** 2
                    +
                    (y1 - y2) ** 2
                )

                ds.append(d)

            ds.sort()

            all_ds.append(
                (
                    tuple(ds),
                    ind
                )
            )

        all_ds.sort(
            key=lambda x: x[0]
        )

        rem = all_ds[0][1]

        arc.remove(rem)

    return arc


def update_arc(
    pop,
    arc,
    size
):
    comb = pop + arc

    spea2_fitness(comb)

    new_arc = [
        ind
        for ind in comb
        if ind.fit < 1.0
    ]

    if len(new_arc) < size:

        rest = [
            ind
            for ind in comb
            if ind not in new_arc
        ]

        rest.sort(
            key=lambda x: x.fit
        )

        n = (
            size - len(new_arc)
        )

        new_arc.extend(
            rest[:n]
        )

    elif len(new_arc) > size:

        new_arc = truncate(
            new_arc,
            size
        )

    return new_arc


def spea_tour(arc, rng):
    a, b = rng.sample(
        arc,
        2
    )

    if a.fit < b.fit:
        return a

    if b.fit < a.fit:
        return b

    return rng.choice(
        [a, b]
    )


def spea2(
    p,
    pop_size,
    max_evals,
    cx_p,
    mut_p,
    seed
):
    if max_evals < pop_size:
        raise ValueError(
            "max_evals must be at least pop_size."
        )

    rng = random.Random(seed)

    pop = make_pop(
        p,
        pop_size,
        rng
    )

    evals = pop_size
    arc = []

    while evals < max_evals:

        arc = update_arc(
            pop,
            arc,
            pop_size
        )

        spea2_fitness(arc)

        off = []

        n_off = min(
            pop_size,
            max_evals - evals
        )

        while len(off) < n_off:

            p1 = spea_tour(
                arc,
                rng
            )

            p2 = spea_tour(
                arc,
                rng
            )

            ch1, ch2 = crossover(
                p1,
                p2,
                cx_p,
                p,
                rng
            )

            ch1 = mutate(
                ch1,
                mut_p,
                p,
                rng
            )

            ch2 = mutate(
                ch2,
                mut_p,
                p,
                rng
            )

            i1 = Ind(ch1)
            eval_ind(i1, p)

            off.append(i1)
            evals += 1

            if (
                len(off) < n_off
                and
                evals < max_evals
            ):
                i2 = Ind(ch2)
                eval_ind(i2, p)

                off.append(i2)
                evals += 1

        pop = off

    arc = update_arc(
        pop,
        arc,
        pop_size
    )

    return unique_front(arc)