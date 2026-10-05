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
