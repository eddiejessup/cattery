import wellmixed
import numpy as np
import argparse
import matplotlib.pyplot as pp
import glob
import os
import matplotlib as mpl

Species = wellmixed.Species


def prefs(species):
    return np.array([s.pref for s in species])


def sigs(species):
    return np.array([s.sig for s in species])


def adaptedness(species, traits):
    return np.abs(prefs(species) - traits)


def names(species):
    return [s.name for s in species]


def analyse(stat_fname, dyns, animated=False):
    stat = np.load(stat_fname)
    traits = stat['traits']
    m = len(traits)

    locs = range(m)

    if animated:
        pp.ion()
        pp.show()
        fig_dist = pp.figure()
        ax_dist = fig_dist.gca()
        ax_dist.set_ylim(0.0, 1.0)
        ax_dist.scatter(locs, traits, c='k', marker='x')
        p = ax_dist.scatter(locs, len(locs) * [np.nan])

    ts, adaptednesses, specialisations = [], [], []
    for dyn_fname in dyns:
        dyn = np.load(dyn_fname)
        species = dyn['species']
        t = dyn['t']
        ts.append(t)
        adaptednesses.append(np.mean(adaptedness(species, traits)))
        specialisations.append(1.0 / np.mean(np.log(sigs(species))))

        p.set_offsets(np.array([locs, prefs(species)]).T)

        ns = np.array(names(species))
        cis = np.array(256 * ns / ns.max(), dtype=np.int)

        c = mpl.cm.jet(cis)
        p.set_color(c)

        fig_dist.canvas.draw()
        print(sigs(species))

    ts = np.array(ts)

    pp.plot(ts, adaptednesses)
    pp.xlabel('t')
    pp.ylabel('Adaptedness')
    pp.title(
        'Mean absolute difference between species preference and environment trait')
    pp.show()

    pp.plot(ts, specialisations)
    pp.xlabel('t')
    pp.ylabel('Specialisation')
    pp.title('Reciprocal of mean gaussian width of all species in environment')
    pp.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse ecosystem dynamics')
    parser.add_argument('dir')
    args = parser.parse_args()

    stat_fname = os.path.join(args.dir, 'static.npz')
    dyns = glob.glob(os.path.join(args.dir, 'dyn', '*.npz'))

    analyse(stat_fname, dyns, True)
