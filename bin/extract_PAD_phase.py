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
import matplotlib.pyplot as plt
import pandas as pd


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


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


def trim_dataframe_to_sb(Psi_phi, sb, Ip):
    """given a dataframe containing the photoelectron momenta, select only the
    energies which lie within the given sideband. Assumes the photon energy is
    0.06 a.u and the sideband is 0.01 a.u wide"""
    photon_energy = 0.06  # in a.u.
    sb_width = 0.01       # sideband summed over the range sb_energy ± sb_width
    sb_energy = sb*photon_energy  # energy of the sideband in a.u
    sb_lo = sb_energy - sb_width
    sb_hi = sb_energy + sb_width

    momenta = [float(x) for x in Psi_phi.columns]
    energies = mom_to_energy(Ip, momenta)
    filt = [(float(x) > sb_lo and float(x) < sb_hi) for x in energies]
    Psi_phi = Psi_phi.loc[:, filt]
    return Psi_phi


def test_func(x, a, b, c, d):
    """function with which to fit the sideband oscillation"""
    return a * np.cos(b * x + c) + d


def getPhase(data, p0=[0, 2, 0, 0]):
    """
    fit the data and extract the phase. Use the parameters from the previous
    fit (p0) as the starting point for the fit.

    Parameters
    ==========
    data: np.array of length 16
        sideband yield as a function of time delay
    p0: list of floats of length 4
        initial values for curve fit: [amplitude, frequency, phase, background]
    """
    from scipy.optimize import curve_fit

    # if the phase is getting large and positive, shift the starting point for
    # the next fit to keep things from getting stuck
    if p0[2] > 1.2:
        p0[2] = p0[2] - np.pi

    bounds = ([-np.inf, 1.99, -1.5*np.pi, -np.inf],
              [np.inf, 2.01, 1.5*np.pi, np.inf])

    phase_delays = [i*np.pi/8 for i in range(16)]

    params, params_covariance = curve_fit(test_func, np.array(phase_delays),
                                          data, p0=p0,
                                          bounds=bounds,
                                          maxfev=1e8, ftol=1e-14)
    return params


def extract_phase(Psi_phi, refangle):
    """
    extract the sideband phase along each radial direction (0-360 degrees).
    Then subtract the phase at refangle to give.
    """

    Psi = np.transpose(Psi_phi.values)

    yield14 = []
    # extract the sideband yield at each angle for each time delay
    for ii in range(16):
        tempyield = []
        tempsi = Psi[:, ii*360:(ii+1)*360]
        for jj in range(360):
            tempyield.append(np.sum(tempsi[:, jj]))

        yield14.append(tempyield)

    y14 = np.array(yield14)
    maxyield = 0
    for ii in range(360):
        maxyield = max(maxyield, sum(y14[:, ii]))

    phase = []
    ratio = []

    p0 = [0, 2, 0, 0]
    for ii in range(0, 360):
        p0 = getPhase(y14[:, ii], p0=p0)
        phase.append(1/np.pi*p0[2])
        ratio.append(sum(y14[:, ii]) / maxyield)

    p0 = getPhase(y14[:, 0], p0=p0)
    # recalculate the phase at 0 degrees using the previous fit parameters
    phase[0] = (1/np.pi*p0[2])
    refphase = phase[refangle]
    phase = [p-refphase for p in phase]
    return phase, ratio


def plot_phase(phi, ratio, args):
    x = np.linspace(0, 2*np.pi, 360)
    if args["polar"]:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for ang, phs, rat in zip(x, phi, ratio):
            ax.plot(ang, phs, '.', color=lighten_color('b', 2*rat))
        ax.set_theta_zero_location("S")
        ax.set_ylim([-1.2, 0.4])
        plt.title('$\Theta_T =$' + f'{args["angle"]}°')
        plt.savefig(f'{args["angle"]}')
        plt.show()
    elif args["plot"]:
        plt.plot(x, phi)
        plt.xlabel("θ(°)")
        plt.ylabel("phase (π radians)")
        plt.show()


def PADphase():
    args = read_command_line()

    Psi_phi = pd.read_csv(args['file'], index_col=0)
    Psi_phi = trim_dataframe_to_sb(Psi_phi, args['sb'], args['ip'])

    phi, rat = extract_phase(Psi_phi, args["angle"])
    ratio = [ii/max(rat) for ii in rat]
    x = np.linspace(0, 2*np.pi, 360)
    np.savetxt(args["output"], np.column_stack((x, phi)))

    plot_phase(phi, ratio, args)


def plot_momentum(Psi, momenta):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    phi = np.linspace(0, 360, num=360, endpoint=True)
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


def PADamp():
    args = read_command_line()

    fullPsi = pd.read_csv(args['file'], index_col=0)
    momenta = [float(x) for x in fullPsi.columns]

    Psi = select_dist(fullPsi, sel=0)
    plot_momentum(Psi, momenta)


def integrateOverAngle(Psi):
    nsum = 0
    for angle in range(360):
        nsum += Psi[:, angle]
    nsum *= np.pi/180
    return nsum


def plot_rabbit(matdat, energies):
    from matplotlib.image import NonUniformImage
    xaxis = np.linspace(0, 2*np.pi, 16)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = NonUniformImage(ax, interpolation='nearest')
    im.set_data(xaxis, energies*27.212, matdat)
    ax.images.append(im)
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 20)
    ax.set_xlabel('phase delay IR-XUV (rad)')
    ax.set_ylabel('Photon Energy (eV)')
    plt.show()


def rabbitt():
    args = read_command_line()
    fullpsi = pd.read_csv(args['file'], index_col=0)
    Psi_phi = np.transpose(fullpsi.values)
    momenta = [float(x) for x in fullpsi.columns]
    energies = mom_to_energy(0.0, momenta)
    matdat = np.zeros((len(energies), 16))
    for td in range(16):
        matdat[:, td] = integrateOverAngle(Psi_phi[:, td*360:(td+1)*360])
    plot_rabbit(matdat, energies)



if __name__ == "__main__":
    rabbitt()
