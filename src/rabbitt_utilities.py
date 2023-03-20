"""
Utilities for working with photoelectron angular momentum distributions from
RMT-calculations and experiment as part of the "Atomic partial wave meter by
attosecond coincidence metrology" paper by W. Jiang et al. Nat. Comms. 2022.

All code written by Andrew C. Brown 2022.

Note that photon energy is hardcoded as 0.06 a.u.

For certain functions, the ionisation potential in a.u needs to be provided as
an input argument (--ip). For others, the sideband index needs to be specified.
The parameters for the figures in the paper are:
For Argon: Ip = 0.579 a.u. sb = 14
For Neon: Ip = 0.793 a.u. sb = 18
For Helium: Ip = 0.904 a.u. sb =18
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

photon_energy = 0.06
debug_yield = False  # set to true to see sideband yields
debug_fit = []  # list of angles for which we want to view the fits.

momentum_file_format = """
The momentum file must be in comma-separated-variable format.

Columns from left to right are increasing momentum

The first line should contain the momentum values (assumed to be in a.u.)

The next 360 lines contain the yield at each of 360 emission angles for the
first time delay

The next 360 lines contain the yield at each of 360 emission angles for the
second time delay

and so on...

Thus the file should have 360*16 + 1 = 5761 lines.

The utility format_csv is provided to fix this. (format_csv --help)
"""


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
                        default="dontsaveme.txt")
    parser.add_argument('-a', '--angle',
                        help="skew angle", default=None)
    parser.add_argument('-i', '--ip', type=float,
                        help="ionisation energy in a.u. (default He)",
                        default=0.904)
    parser.add_argument('-s', '--sb', type=int,
                        help="sideband index",
                        default=18)
    args = vars(parser.parse_args())

    with open(args['file'], 'r') as f:
        lines = f.readlines()
        if ("," not in lines[1]) or ((len(lines)-1) % 16 != 0):
            raise IOError(
                f"""Momentum file formatted incorrectly.
                {momentum_file_format}""")

    return args


def mom_to_energy(Ip, mom):
    """ given an ionisation energy and a list of equally spaced photoelectron
    momentum values, convert those values to photoelectron energies. All values
    assumed to be in atomic units"""
    factor = mom[1] - mom[0]
    zeniths = np.zeros(len(mom))
    for ll in range(0, len(mom)):
        zeniths[ll] = Ip+0.5*ll*ll*factor*factor
    return zeniths


def trim_dataframe_to_sb(Psi_phi, sb, Ip, refangle=0):
    """given a dataframe containing the photoelectron momenta, select only the
    energies which lie within the given sideband. Assumes the sideband is
    0.01 a.u wide
    Parameters
    ==========
    Psi_phi: pd.DataFrame
        dataframe containing the photoelectron momenta. Each column corresponds
        to a specific momentum.
    sb: int
        the sideband index of interest
    Ip: float
        the ionisation energy. This is required to ensure the energy is
        calculated relative to the neutral ground state.
    refangle: int
        skew angle ΘT between the XUV-APT and IR

    Returns
    =======
    Psi_Phi: pd.DataFrame
        dataframe containing the photoelectron momenta limited to energies
        within the sideband of interest.
    """
    sb_width = 0.01       # sideband summed over the range sb_energy ± sb_width
    sb_energy = sb*photon_energy  # energy of the sideband in a.u
    sb_lo = sb_energy - 2*sb_width
    sb_hi = sb_energy + sb_width

    momenta = [float(x) for x in Psi_phi.columns]
    energies = mom_to_energy(Ip, momenta)
    filt = [(float(x) > sb_lo and float(x) < sb_hi) for x in energies]
    ens = list(filter(lambda x: float(x) >
               sb_lo and float(x) < sb_hi, energies))

    Psi_phi = Psi_phi.loc[:, filt]
    if debug_yield:
        yy = []
        for ii in range(16):
            yy.append(np.sum(Psi_phi.iloc[ii*360+refangle]))
            plt.plot(ens, Psi_phi.iloc[ii*360+refangle])
            plt.xlabel('Photon Energy (a.u.)')
            plt.ylabel('Amplitude (arb. units)')
            plt.title(f'sideband {sb} yield for time delay {ii+1} of 16')
            plt.show()
        plt.plot(np.arange(16), yy)
        plt.xlabel('Time-delay index')
        plt.ylabel('Sideband yield')
        plt.title(f'Sideband {sb} yield as a function of time delay')
        plt.show()
    return Psi_phi


def test_func(x, a, b, c, d):
    """function with which to fit the sideband oscillation"""
    return a * np.cos(b * x + c) + d


def getPhase(data, p0=[0, 2, 0, 0], ang=None):
    """
    fit the data and extract the phase. Use the parameters from the previous
    fit (p0) as the starting point for the fit.

    Parameters
    ==========
    data: np.array of length 16
        sideband yield as a function of time delay
    p0: list of floats of length 4
        initial values for curve fit: [amplitude, frequency, phase, background]

    Returns
    =======
    params : list of length 4
        fit parameters [amplitude, frequency, phase, background]
    """
    from scipy.optimize import curve_fit

    bounds = ([-np.inf, 1.99, -2.0*np.pi, -np.inf],
              [np.inf, 2.01, 2.0*np.pi, np.inf])

    phase_delays = [i*np.pi/8 for i in range(16)]

    params, params_covariance = curve_fit(test_func, np.array(phase_delays),
                                          data, p0=p0,
                                          bounds=bounds,
                                          maxfev=1e8, ftol=1e-14)
    if ang in debug_fit:
        plt.plot(phase_delays, data, 'r.')
        plt.plot(phase_delays, test_func(np.array(phase_delays), *params))
        plt.title(f'fitted sideband yield for angle {ang}')
        plt.show()
    return params


def extract_phase(Psi_phi, refangle=None):
    """
    extract the sideband phase along each radial direction (0-360 degrees).
    Then subtract the phase at refangle to give.

    Parameters
    ==========
    Psi_phi : pd.DataFrame
        dataframe containing the photoelectron momenta limited to energies
        within the sideband of interest.
    refangle : float or int
        skew angle ΘT between the XUV-APT and IR, If provided then the
        phase is the normalized phase shift Δϕ = ϕ(θ) - ϕ(ΘT).

    Returns
    =======
    phase : list of 360 floats
        the sideband phase extracted from the fit to the rabbitt data along
        each emission angle
    relative_yield: list of 360 floats
        the yield as a fraction of the maximum yield of the sideband along each
        emission angle
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
    if abs(maxyield) < 1e-13:
        return None, None

    phase = []
    ratio = []

    p0 = [0, 2, 0, 0]
    for ii in range(0, 360):
        p0 = getPhase(y14[:, ii], p0=p0)
        phase.append(1/np.pi*p0[2])
        ratio.append(sum(y14[:, ii]) / maxyield)

# calculate the phase again using the previous fit parameters
# (helps convergence)
    phase = []
    ratio = []
    for ii in range(0, 360):
        p0 = getPhase(y14[:, ii], p0=p0, ang=ii)
        phase.append(1/np.pi*p0[2])
        ratio.append(sum(y14[:, ii]) / maxyield)
    if refangle:
        refangle = int(refangle)
        refphase = phase[refangle]
        phase = [p-refphase for p in phase]
    relative_yield = [ii/max(ratio) for ii in ratio]
    return phase, relative_yield


def plot_phase(phi, ratio, args):
    """ show the extracted sideband phase as a function of emission angle as
    either a polar or cartesian plot. If polar, weight the colour of the
    plotted phase by the yield.

    Parameters
    ==========
    phi : list of 360 floats
        the sideband phase extracted from the fit to the rabbitt data along
        each emission angle
    ratio: list of 360 floats
        the yield as a fraction of the maximum yield of the sideband along each
        emission angle
    args: dict
        command line arguments used to specify plot parameters
    """
    x = np.linspace(0, 2*np.pi, 360)
    if args["polar"]:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for ang, phs, rat in zip(x, phi, ratio):
            ax.plot(ang, phs, '.', color=lighten_color('b', 2*rat))
        ax.set_theta_zero_location("S")
#        ax.set_ylim([-1.2, 0.4])
        plt.title('$\Theta_T =$' + f'{args["angle"]}°')
        plt.savefig(f'{args["angle"]}')
        plt.show()
    elif args["plot"]:
        plt.plot(x, phi)
        plt.xlabel("θ(°)")
        plt.ylabel("phase (π radians)")
        plt.show()


def PADphase(args):
    """
    Utility for extracting the sideband phase as a function of emission angle
    for angle resolved rabbitt spectra. If command line option -a/--angle is
    used this refers to the skew angle ΘT between the XUV-APT and IR, and the
    phase is then the normalized phase shift Δϕ = ϕ(θ) - ϕ(ΘT).

    Parameters
    ==========
    args: dict
        command line arguments

    Returns
    =======
    phase : float
        extracted phase averaged over all emission angles (used for partial
        wave data where the phase is constant over emission angles)
    """

    Psi_phi = pd.read_csv(args['file'])
    Psi_phi = trim_dataframe_to_sb(Psi_phi, args['sb'], args['ip'])

    phi, ratio = extract_phase(Psi_phi, args["angle"])
    if (phi):
        if args["output"] != "dontsaveme.txt":
            x = np.linspace(0, 2*np.pi, 360)
            np.savetxt(args["output"], np.column_stack((x, phi)))

        plot_phase(phi, ratio, args)
        return np.average(phi)
    else:
        return f"No yield in sideband {args['sb']} for file {args['file']}"


def plot_momentum(Psi, momenta):
    """Show a polar plot of the photoelectron angular momentum distribution.

    Parameters
    ==========
    Psi: np.array of size (:, 360)
        Photoelectron momentum for each emission angle
    momenta: list of floats
        Photoelectron momentum values corresponding to the first axis of Psi
    """

    import matplotlib.pyplot as plt
    from matplotlib import cm
    phi = np.linspace(0, 360, num=360, endpoint=True)
    angle = np.radians(-phi)
    theta, r = np.meshgrid(angle, momenta)
    plt.figure(1, figsize=(8, 9))
    ax = plt.subplot(polar=True)
    ax.set_theta_zero_location("E")
#    lup = 1.01*np.amax(Psi)
    levels = np.linspace(0.0, 0.0011, 200)
    CS = plt.contourf(theta, r, Psi, levels, cmap=cm.jet)
    ax.set_rmax(2.0)
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

    if not (sel):
        Psi = 0
        for ii in range(16):
            Psi += fullPsi[:, ii*360:(ii+1)*360]
    else:
        Psi = fullPsi[:, sel*360:(sel+1)*360]
    return (Psi)


def PADamp(args):
    """Read Photoelectron angular momentum distribution from file and plot it
    """
    fullPsi = pd.read_csv(args['file'], index_col=0)
    momenta = [float(x) for x in fullPsi.columns]

    Psi = select_dist(fullPsi, sel=0)
    plot_momentum(Psi, momenta)


def integrateOverAngle(Psi):
    """Integrate the photoelectron angular momentum distribution over all
    emission angles.
    Parameters
    ==========
    Psi: np.array of size (:, 360)
        Photoelectron momentum for each emission angle

    Returns
    =======
    nsum : np.array of size(:)
        Integrated photoelectron momentum (radial distribution)
    """
    nsum = 0
    for angle in range(180):
        nsum += Psi[:, angle] * np.pi * 2 * np.sin(np.radians(angle))
    nsum *= np.pi/180
    return nsum


def plot_rabbit(matdat, energies):
    """Show a colour plot of the rabbitt spectrum

    Parameters
    ==========
    matdat : np.ndarray of size(num_energies, num_time_delays)
        rabbitt spectrum
    energies: np.array
        photoelectron energies corresponding to axis-0 of matdat
    """
    from matplotlib.image import NonUniformImage
    xaxis = np.linspace(0, 2*np.pi, 16)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = NonUniformImage(ax, interpolation='nearest')
    im.set_data(xaxis, energies*27.212, matdat)
    ax.add_image(im)
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(27.212*energies[0], 27.212*energies[-1])
    ax.set_xlabel('phase delay IR-XUV (rad)')
    ax.set_ylabel('Photon Energy (eV)')
    plt.show()


def rabbitt_phase(momenta, matdat, sb, ip=0):
    """
    extract sideband phase for the angle integrated spectrum

    Parameters
    ==========
    momenta: list of floats
        Photoelectron momentum values corresponding to the first axis of Psi
    matdat : np.ndarray
        of size (num_energies, 16), contains the photoelectron yield at each
        energy for each time delay

    sb : int
        sideband of interest

    ip : float
        ionisation potential (in a.u)

    Returns
    =======
    phase : float
        the sideband phase extracted from the fit to the rabbitt data
    """

    energies = mom_to_energy(ip, momenta)
    for ii, en in enumerate(energies):
        if en > sb*photon_energy-0.01:
            i_lo = ii
            break
    for ii, en in enumerate(energies):
        if en > sb*photon_energy+0.01:
            i_hi = ii
            break
    sbyield = []
    for td in range(16):
        sbyield.append(np.sum(matdat[i_lo:i_hi, td]))

    return getPhase(sbyield)[2]


def rabbitt(args):
    """Read momentum distribution from file, integrate over angle, and plot the
    rabbitt spectrum"""
    fullpsi = pd.read_csv(args['file'], index_col=0)
    Psi_phi = np.transpose(fullpsi.values)
    momenta = [float(x) for x in fullpsi.columns]
    energies = mom_to_energy(0.0, momenta)
    matdat = np.zeros((len(energies), 16))
    for td in range(16):
        matdat[:, td] = integrateOverAngle(Psi_phi[:, td*360:(td+1)*360])
    if args['plot']:
        plot_rabbit(matdat, energies)
    phs = rabbitt_phase(momenta, matdat, args['sb'], args['ip'])
    print(f"Phase for sb{args['sb']} is {phs}")


def pwPhase(args):
    """read partial wave momentum distributions and calculate the absolute or
    relative phase associated with that partial wave"""
    pwPhase = []

    pwPhase.append(PADphase(args))
    if type(pwPhase[0]) == str:
        print(pwPhase[0])
        return

    refwave = input(
        "Enter the reference-wave csv filename (blank for absolute phase): ")
    if (refwave != ""):
        args["file"] = refwave
        pwPhase.append(PADphase(args))
        print(f"Relative phase: {pwPhase[0]-pwPhase[1]}")
    else:
        print(f"Absolute phase: {pwPhase[0]}")


if __name__ == "__main__":
    pass
