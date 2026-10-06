from dataclasses import dataclass


@dataclass
class CFLP:
    nf: int
    nc: int
    caps: list
    fc: list
    dem: list
    ac: list


def load_data(path):
    with open(path, "r") as f:
        vals = f.read().split()

    i = 0

    nf = int(vals[i])
    i += 1

    nc = int(vals[i])
    i += 1

    caps = []
    fc = []

    for _ in range(nf):
        caps.append(float(vals[i]))
        i += 1

        fc.append(float(vals[i]))
        i += 1

    dem = []
    ac = []

    for _ in range(nc):
        dem.append(float(vals[i]))
        i += 1

        row = []

        for _ in range(nf):
            row.append(float(vals[i]))
            i += 1

        ac.append(row)

    return CFLP(
        nf=nf,
        nc=nc,
        caps=caps,
        fc=fc,
        dem=dem,
        ac=ac
    )


def loads(ch, p):
    ls = [0.0] * p.nf

    for c, f in enumerate(ch):
        ls[f] += p.dem[c]

    return ls


def feasible(ch, p):
    if len(ch) != p.nc:
        return False

    for f in ch:
        if f < 0 or f >= p.nf:
            return False

    ls = loads(ch, p)

    for f in range(p.nf):
        if ls[f] > p.caps[f] + 1e-9:
            return False

    return True


def make_ch(p, rng):
    if sum(p.dem) > sum(p.caps):
        raise ValueError(
            "Total demand is larger than total capacity."
        )

    cs = list(range(p.nc))

    for _ in range(1000):
        rem = p.caps.copy()
        ch = [-1] * p.nc
        used = set()

        rng.shuffle(cs)

        cs.sort(
            key=lambda c: -p.dem[c]
        )

        ok = True
        mode = rng.randrange(3)

        for c in cs:
            d = p.dem[c]

            opts = [
                f
                for f in range(p.nf)
                if rem[f] >= d
            ]

            if not opts:
                ok = False
                break

            if mode == 0:
                opts.sort(
                    key=lambda f: p.ac[c][f]
                )

                k = min(4, len(opts))
                f = rng.choice(opts[:k])

            elif mode == 1:
                opts.sort(
                    key=lambda f: (
                        0 if f in used else p.fc[f],
                        p.ac[c][f]
                    )
                )

                k = min(4, len(opts))
                f = rng.choice(opts[:k])

            else:
                f = rng.choice(opts)

            ch[c] = f
            rem[f] -= d
            used.add(f)

        if ok:
            return ch

    raise RuntimeError(
        "Could not create a feasible solution."
    )


def repair(ch, p, rng):
    ch = ch.copy()

    for c in range(p.nc):
        if ch[c] < 0 or ch[c] >= p.nf:
            ch[c] = rng.randrange(p.nf)

    ls = loads(ch, p)

    max_try = p.nc * p.nf * 3

    for _ in range(max_try):

        over = [
            f
            for f in range(p.nf)
            if ls[f] > p.caps[f] + 1e-9
        ]

        if not over:
            return ch

        best = None
        best_v = float("inf")

        used = set(ch)

        for old_f in over:

            cs = [
                c
                for c in range(p.nc)
                if ch[c] == old_f
            ]

            for c in cs:
                d = p.dem[c]

                for new_f in range(p.nf):

                    if new_f == old_f:
                        continue

                    if (
                        ls[new_f] + d
                        > p.caps[new_f] + 1e-9
                    ):
                        continue

                    v = (
                        p.ac[c][new_f]
                        - p.ac[c][old_f]
                    )

                    if new_f not in used:
                        v += 0.01 * p.fc[new_f]

                    v += rng.random() * 1e-6

                    if v < best_v:
                        best_v = v
                        best = (
                            c,
                            old_f,
                            new_f
                        )

        if best is None:
            break

        c, old_f, new_f = best

        d = p.dem[c]

        ch[c] = new_f

        ls[old_f] -= d
        ls[new_f] += d

    if feasible(ch, p):
        return ch

    return make_ch(p, rng)


def eval_obj(ch, p):
    if not feasible(ch, p):
        raise ValueError(
            "Objectives cannot be calculated "
            "for an infeasible solution."
        )

    open_f = set(ch)        # Open facilities

    f1 = sum(
        p.fc[f]
        for f in open_f
    )

    f2 = sum(
        p.ac[c][f]
        for c, f in enumerate(ch)
    )

    return f1, f2