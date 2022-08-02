"""
Utility for extracting the sideband phase as a function of emission angle for
angle resolved rabbitt spectra. If command line option -a/--angle is used this
refers to the skew angle ΘT between the XUV-APT and IR, and the phase is then
the normalized phase shift Δϕ = ϕ(θ) - ϕ(ΘT).

Parameters for the figures in the paper:
For Argon: Ip = 0.579 a.u. sb = 14
For Neon: Ip = 0.793 a.u. sb = 18
For Helium: Ip = 0.904 a.u. sb =18
"""
import numpy as np
import pandas as pd


def read_command_line():
    from argparse import ArgumentParser as AP
    parser = AP()
    parser.add_argument('file',
                        help="OuterWave_momentum file")
    parser.add_argument('-p', '--plot',
                        help="display a plot of the extracted phase",
                        default=False, action='store_true')
    parser.add_argument('--polar',
                        help="use polar instead of linear plot ",
                        default=False, action='store_true')
    parser.add_argument('-o', '--output', type=str,
                        help="file for phase to be stored",
                        default="phase.txt")
    parser.add_argument('-a', '--angle', type=int,
                        help="skew angle", default=0)
    parser.add_argument('-i', '--ip', type=float,
                        help="ionisation energy in a.u. (default He)",
                        default=0.904)
    parser.add_argument('-s', '--sb', type=int,
                        help="sideband index",
                        default=18)

    return vars(parser.parse_args())


def mom_to_energy(Ip, mom):
    """ given an ionisation energy and a list of equally spaced photoelectron
    momentum values, convert those values to photoelectron energies. All values
    assumed to be in atomic units"""
    factor = mom[1] - mom[0]
    zeniths = np.zeros(len(mom))
    for ll in range(0, len(mom)):
        zeniths[ll] = Ip+0.5*ll*ll*factor*factor
    return zeniths


def plot_momentum(Psi):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    phi = np.linspace(0, 360, num=360, endpoint=False)
    angle = np.radians(-phi)
    theta, r = np.meshgrid(angle, momenta)
    plt.figure(1, figsize=(8, 9))
    ax = plt.subplot(polar=True)
    ax.set_theta_zero_location("E")
    lup = 1.01*np.amax(Psi)
    levels = np.linspace(0.0, lup, 200)
    CS = plt.contourf(theta, r, Psi, levels, cmap=cm.jet)
    ax.set_rmax(1.0)
    rlabels = ax.get_ymajorticklabels()
    for label in rlabels:
        label.set_color('white')
    ax.set_rlabel_position(135)
    ax.tick_params(labelsize=10, direction='in')
    thetaticks = np.arange(0, 360, 45)
    ax.set_thetagrids(thetaticks)  # ,frac=1.15)
    cbar = plt.colorbar(CS)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('$p_{y}$', size=20)
    ax.set_ylabel('$p_{z}$', size=20)
    ax.yaxis.set_label_coords(-0.05, 0.52)
    plt.show()


def select_dist(fullPsi, sel=None):
    """ If selection sel=None, then overlay all time delays. Otherwise select
    specific time delay(sel=0,..15) chooses one of the sixteen time delays
    """
    fullPsi = np.transpose(fullPsi.values)

    if not(sel):
        Psi = 0
        for ii in range(16):
            Psi += fullPsi[:, ii*360:(ii+1)*360]
    else:
        Psi = fullPsi[:, sel*360:(sel+1)*360]
    return(Psi)


args = read_command_line()

fullPsi = pd.read_csv(args['file'], index_col=0)
momenta = [float(x) for x in fullPsi.columns]
energies = mom_to_energy(args['ip'], momenta)

Psi = select_dist(fullPsi, sel=0)
plot_momentum(Psi)
