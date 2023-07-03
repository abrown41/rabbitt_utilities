"""
Utilities for working with photoelectron angular momentum distributions from
RMT-calculations and experiment as part of the "Atomic partial wave meter by
attosecond coincidence metrology" paper by W. Jiang et al. Nat. Comms. 2022.

Code written by Andrew C. Brown 2022, 
updated by Luke F. Roantree 2023, 
adding the following functionality;
> account for pondermotive shifting
> isolate resonance imprints in spectral phase
> energy-resolved phase
> account for 4-omega oscillations from 4-photon processes
> arbitrary number of XUV-NIR delays
> enable 'manual shift' of estimated sideband interval, with plot to confirm location
> uncertainty analysis and visulaisation

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
#from matplotlib import use
#use('agg')
import pandas as pd

photon_energy = 0.05875
debug_yield = True # set to true to see sideband yields
debug_fit = [0, 20, 55, 75, 90]  # list of angles for which we want to view the fits.

momentum_file_format = """
The momentum file must be in comma-separated-variable format.

Columns from left to right are increasing momentum

The first line should contain the momentum values (assumed to be in a.u.)

The next 360 lines contain the yield at each of 360 emission angles for the
first time delay

The next 360 lines contain the yield at each of 360 emission angles for the
second time delay

and so on...

Thus the file should have 360*num_delays + 1 = 5761 lines for default 16 delays.

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
    parser.add_argument('-l', '--label', type=str,
                        help="filename for angular phase image",
                        default=None)
    parser.add_argument('-a', '--angle',
                        help="skew angle", default=None)
    parser.add_argument('-i', '--ip', type=float,
                        help="ionisation energy in a.u. (default He)",
                        default=0.904)
    parser.add_argument('-s', '--sb', type=int,
                        help="sideband index",
                        default=18)
    parser.add_argument('-r', '--res', type=int,
                        help="resonance (4=4p, 5=5p, default 0)",
                        default=0)
    parser.add_argument('-n', '--num_delays', type=int,
                        help="number of time delays used (default 16)",
                        default=16)
    parser.add_argument('--intensity', type=str,
                        help="Intensity used (W/cm^2), format 'XEY' (default 1E12)",
                        default="1E12")
    parser.add_argument('-m', '--manual_shift', type=float, 
                        help='Check VERIFY/yields.png - manually adjust SB by shifting <num>eV',
                        default=0)
    parser.add_argument('-e', '--energy', type=float, 
                        help='driving IR photon energy (au), default 0.6',
                        default=0.05875)
    
    args = vars(parser.parse_args())
    
    with open(args['file'], 'r') as f:
        lines = f.readlines()
        if ("," not in lines[1]) or ((len(lines)-1) % args['num_delays'] != 0):
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

def photon_energy_au_to_wavelength_nm(photon_energy):
    """
    Converts the global photon_energy parameter to its corresponding
    wavelength in nm.
    """
    return 1239.8/(photon_energy*27.21138)

def get_pondermotive_shift_eV(intensity, photon_energy):
    """
    Given an intensity (string, form XEY), use the globally set photon_energy
    to calculate the pondermotive energy in eV.
    We will then assume this translates well to a shift in the ground state
    energy, causing the sidebands to 'shift' by this value.
    Paramters
    =========
    intensity: string 
        form 'XEY', e.g. '2E12' for 2x10^12 W/cm^2

    Returns
    =======
    UP: float
        Pondermotive Energy (eV)
    """
    In = float(intensity.split('E')[0])*10**int(intensity.split('E')[1])
    In /= 1e15
    W = photon_energy_au_to_wavelength_nm(photon_energy)
    UP = W**2 * In * 9.33738/100000
    return UP

def trim_dataframe_to_sb(Psi_phi, sb, Ip, refangle=0, fourp=0, 
                         fivep=0, num_delays=16, intensity='1E12', 
                         manual_shift=0, photon_energy=0.05875):
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
        skew angle ΘT between the XUV-APT and IR.
    fourp: bool
        whether or not to select only the spectral range encompassing the 1s4p resonance imprint (He SB16 only).
    fivep: bool
        whether or not to select only the spectral range encompassing the 1s5p resonance imprint (He SB16 only).
    num_delays: int
        how many XUV-NIR delayed spectra are present in the input data. Assumes equally spaced from 0-T_{NIR}.
    intensity: string
        The intensity of the laser field, used to calculate the pondermotive shift. 
        Should be in form XEY, e.g. 1E12 for 1x10^(12) W/cm^2.
    manual_shift: float
        After checking ./VERIFY/yields.png, shift the calculated sideband interval by some small amount (eV)
        to better align with the true sideband location. 
        e.g. -0.01 to shift the spectral interval down by 0.01eV.

    Returns
    =======
    Psi_Phi: pd.DataFrame
        dataframe containing the photoelectron momenta limited to energies
        within the sideband of interest.
    """

    sb_width = 0.01
    pond_shift = get_pondermotive_shift_eV(intensity, photon_energy)/27.21138
    print(f'Accounting fo pondermotive shift of: {pond_shift:.4e}')
    sb_energy = sb*photon_energy - pond_shift + (manual_shift/27.21138)
    if not fivep:
        sb_lo = sb_energy - sb_width
    else:
        sb_lo = sb_energy 
    if not fourp: 
        sb_hi = sb_energy + sb_width
    else:
        sb_hi = sb_energy
    print(f'SB limits: {sb_lo:.4f} {sb_hi:.4f}')
    momenta = [float(x) for x in Psi_phi.columns]
    energies = mom_to_energy(Ip, momenta)
    filt = [(float(x) > sb_lo and float(x) < sb_hi) for x in energies]
    ens = list(filter(lambda x: float(x) >
               sb_lo and float(x) < sb_hi, energies))

    Psi_phi = Psi_phi.loc[:, filt]
    if debug_yield:
        yy = []
        plt.figure()
        for ii in range(num_delays):
            yy.append(np.sum(Psi_phi.iloc[ii*360+refangle]))
            plt.plot([27.21138*e for e in ens], Psi_phi.iloc[ii*360+refangle])
        plt.xlabel('Photon Energy (a.u.)')
        plt.ylabel('Amplitude (arb. units)')
        plt.title(f'sideband {sb} yield for {num_delays} delays')
        plt.savefig('./VERIFY/yields.png', dpi=250)
        plt.close()
        plt.figure()
        plt.plot(np.arange(num_delays), yy)
        plt.xlabel('Time-delay index')
        plt.ylabel('Sideband yield')
        plt.title(f'Sideband {sb} yield as a function of time delay')
        plt.savefig('./VERIFY/all.png', dpi=250)
        plt.close()
    return Psi_phi


def test_func(x, a1, c1, a2, c2, d, e):
    """function with which to fit the sideband oscillation"""
    return a1 * np.cos(2*x + c1) + a2 * np.cos(4*x + c2) + d + e*x 

def test_func_onecos(x, a, c, d, e, is_4omega=False):
    """function for visualising fitted values"""
    if is_4omega:
        return a * np.cos(4*x + c) + d + e*x 
    else:
        return a * np.cos(2*x + c) + d + e*x 

def getPhase(data, p0=None, ang=None, num_delays=16):
    """
    fit the data and extract the phase. Use the parameters from the previous
    fit (p0) as the starting point for the fit.

    Parameters
    ==========
    data: np.array of length 16
        sideband yield as a function of time delay
    p0: list of floats of length 4
        initial values for curve fit: [amplitude, frequency, phase, background]
    ang: int
        emission angle at which phases are being extracted. Only used for debugging
        or plotting purposes. Leave unset if extracting angle-integrated phase.
    num_delays: int
        The number of XUV-NIR delayed spectra present in the input data

    Returns
    =======
    params : list of length 6 (floats)
        fit parameters [amplitude omega2, phase omega2, amplitude omega4, 
                        phase omega4, background drift, background constant]
    uncertainties: list of length 6 (floats)
        standard deviations of each fitted parameter
    """
    from scipy.optimize import curve_fit

    scale = np.mean(data)
    data_fits = data.copy()
    data_fits /= scale

    if p0 is not None:
        p0 = [2*np.mean(data_fits), 0, 0.1*np.mean(data_fits), 0, np.mean(data_fits)/5, 0]

    bounds = ([0, -2*np.pi, 0, -2*np.pi, -np.inf, -np.inf ],
              [np.inf, 2.0*np.pi,np.inf, 2*np.pi, np.inf, np.inf])

    phase_delays = [i*2*np.pi/num_delays for i in range(num_delays)] 

    params, params_covariance = curve_fit(test_func, 
                                          np.array(phase_delays),
                                          data_fits, 
                                          p0=p0,
                                          bounds=bounds,
                                          maxfev=1e8, 
                                          ftol=1e-14,
                                          method='trf')
    if ang in debug_fit:
        plt.figure()
        print(ang, 'phase 2w std:', np.sqrt(params_covariance[1,1]))
        print(ang, 'phase 4w std:', np.sqrt(params_covariance[3,3]))
        plt.plot(phase_delays, data, 'r.', label='raw')
        plt.plot(phase_delays, test_func(np.array(phase_delays), *params)*scale, label='full fit')
        plt.plot(phase_delays, test_func_onecos(np.array(phase_delays), *[params[i] for i in (0,1,4,5)])*scale, alpha=0.4, label='2w fit + bg')
        plt.plot(phase_delays, test_func_onecos(np.array(phase_delays), *[params[2],params[3],params[4],params[5],True])*scale, alpha=0.4, label='4w fit + bg')
        plt.legend()
        plt.title(f'fitted sideband yield for angle {ang}\n2-,4-omega phases & signal ratio: {params[1]:.2f}, {params[3]:.2f}, {params[0]/params[2]:.2f}')
        plt.savefig(f'debug_{ang}.png', dpi=250)
        print(f"(2 omega / 4 omega) ratio = {params[0]/params[2]:.3f}")

        plt.figure()
        plt.plot(phase_delays, data - test_func(np.array(phase_delays), *params)*scale, 'x', label='residuals')
        plt.title(f'residuals for {ang}deg')
        plt.legend()
        #plt.savefig(f'debugres_{ang}.png', dpi=250)
    return params, np.sqrt(np.diag(params_covariance))


def extract_phase(Psi_phi, refangle=None, num_delays=16):
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
    num_delays: int
        The number of XUV-NIR delayed spectra present in the input data.

    Returns
    =======
    phase : list of 360 floats
        the sideband 2-omega phase extracted from the fit to the rabbitt data along
        each emission angle
    relative_yield: list of 360 floats
        the yield as a fraction of the maximum yield of the sideband along each
        emission angle
    phase_4omega: list of 360 floats
        as for 'phase', but instead relating to the 4-omega phase
    """

    Psi = np.transpose(Psi_phi.values)

    yield14 = []
    angle_int_yield = np.zeros(Psi.shape[0])
    # extract the sideband yield at each angle for each time delay
    for ii in range(num_delays):
        tempyield = []
        temp_angle_int_yield = np.zeros(Psi.shape[0])
        tempsi = Psi[:, ii*360:(ii+1)*360]
        for jj in range(360):
            tempyield.append(np.sum(tempsi[:, jj]))
            temp_angle_int_yield += np.sin(jj*np.pi/180.0)*np.sum(tempsi[:,jj])
        yield14.append(tempyield)
        angle_int_yield += temp_angle_int_yield/num_delays
    y14 = np.array(yield14)
    maxyield = 0
    for ii in range(360):
        maxyield = max(maxyield, sum(y14[:, ii]))
    if abs(maxyield) < 1e-13:
        return None, None

    phase = []
    phase_4omega = []
    ratio = []

    p0 = [0, 2, 0, 0, 0]
    for ii in range(0, 360):
        p0,_ = getPhase(y14[:, ii], p0=p0, num_delays=num_delays)
        phase.append(1/np.pi*p0[1])
        phase_4omega.append(1/np.pi*p0[3])
        ratio.append(sum(y14[:, ii]) / maxyield)

# calculate the phase again using the previous fit parameters
# (helps convergence)
    phase = []
    phase_4omega = []
    ratio = []
    for ii in range(0, 360):
        p0,_ = getPhase(y14[:, ii], p0=p0, ang=ii, num_delays=num_delays)
        phase.append(1/np.pi*p0[1])
        phase_4omega.append(1/np.pi*p0[3])
        ratio.append(sum(y14[:, ii]) / maxyield)
    if refangle:
        refangle = int(refangle)//2
        refphase = phase[refangle]
        refphase_4omega = phase_4omega[refangle]
        phase = [p-refphase for p in phase]
        phase_4omega = [p-refphase_4omega for p in phase_4omega]
    relative_yield = [ii/max(ratio) for ii in ratio]
    return phase, relative_yield, phase_4omega


def plot_phase(phi, ratio, args, is_4omega=False):
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
    is_4omega: bool
        whether or not the figure generated presents the 4-omega spectral phase.
        Only used for determining the filename.
    """
    x = np.linspace(0, 2*np.pi, 360)
    if args["polar"]:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        #for ang, phs, rat in zip(x, phi, ratio):
        #    ax.plot(ang, phs, 'b.') #, color=lighten_color('b', 2*rat))
        ax.plot(x, phi, 'b.', zorder=1)
        rad = np.linspace(1.15, 1.20, 2)
        r, th = np.meshgrid(rad, x)
        z = np.array([ratio, ratio]).T
        plt.pcolormesh(th, r, z, zorder=2)
        ax.set_theta_zero_location("N")
        ax.set_ylim([-1, 1.2])
        ax.grid(True, zorder=3)
        plt.title('$\Theta_T =$' + f'{args["angle"]}°')
        filename = args["label"] if (args["label"] is not None) else f'{args["angle"]}'
        if is_4omega:
            filename = filename+'_4omega'
        else:
            filename = filename+'_2omega'
        plt.savefig(filename + '.png', dpi=275)
        #plt.show()
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
    phi_2omega : float
        extracted 2-omega phase averaged over all emission angles (used for partial
        wave data where the phase is constant over emission angles)
    phi_4omega : float
        as for phi_2omega, but relating to the 4-omega phase
    """

    fourp, fivep = False, False
    if args['res'] == 4:
        fourp = True
    elif args['res'] == 5:
        fivep = True

    Psi_phi = pd.read_csv(args['file'])
    Psi_phi = trim_dataframe_to_sb(Psi_phi, args['sb'], args['ip'], fourp=fourp, fivep=fivep, 
                                   num_delays=args['num_delays'], intensity=args['intensity'], 
                                   manual_shift=args['manual_shift'], photon_energy=args['energy'])
    phi_2omega, ratio, phi_4omega = extract_phase(Psi_phi, refangle=args["angle"], num_delays=args['num_delays'])
    if (phi_2omega):
        if args["output"] != "dontsaveme.txt":
            x = np.linspace(0, 2*np.pi, 360)
            np.savetxt(args["output"]+'2', np.column_stack((x, phi_2omega)))
            np.savetxt(args["output"]+'4', np.column_stack((x, phi_4omega)))

        plot_phase(phi_2omega, ratio, args, is_4omega=False)
    if phi_4omega:
        plot_phase(phi_4omega, ratio, args, is_4omega=True)
    if (phi_2omega and phi_4omega):
        return np.average(phi_2omega), np.average(phi_4omega)
    else:
        raise ValueError(f"No yield in sideband {args['sb']} for file {args['file']}")


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


def select_dist(fullPsi, sel=None, num_delays=16):
    """ If selection sel=None, then overlay all time delays. Otherwise select
    specific time delay(sel=0,..(num_delays-1)) chooses one of the <num_delays> time delays
    """
    fullPsi = np.transpose(fullPsi.values)

    if not (sel):
        Psi = 0
        for ii in range(num_delays):
            Psi += fullPsi[:, ii*360:(ii+1)*360]
    else:
        Psi = fullPsi[:, sel*360:(sel+1)*360]
    return (Psi)


def PADamp(args):
    """Read Photoelectron angular momentum distribution from file and plot it
    """
    fullPsi = pd.read_csv(args['file'], index_col=0)
    momenta = [float(x) for x in fullPsi.columns]

    Psi = select_dist(fullPsi, sel=0, num_delays=args['num_delays'])
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


def plot_rabbit(matdat, energies, num_delays=16):
    """Show a colour plot of the rabbitt spectrum

    Parameters
    ==========
    matdat : np.ndarray of size(num_energies, num_time_delays)
        rabbitt spectrum
    energies: np.array
        photoelectron energies corresponding to axis-0 of matdat
    num_delays: int
        Number of XUV-NIR delayed spectra present in matdat
    """
    from matplotlib.image import NonUniformImage
    xaxis = np.linspace(0, 2*np.pi, num_delays)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = NonUniformImage(ax, interpolation='nearest')
    im.set_data(xaxis, energies*27.212, matdat)
    ax.add_image(im)
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(27.212*energies[0], 27.212*energies[-1])
    ax.set_xlabel('phase delay IR-XUV (rad)')
    ax.set_ylabel('Photon Energy (eV)')
    plt.show()


def rabbitt_phase(momenta, matdat, sb, ip=0, num_delays=16, photon_energy=0.05875):
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

    num_delays: int
        number of XUV-NIR delayed spectra present in matdat

    Returns
    =======
    phase : float
        the fitted parameters extracted the rabbitt data

    uncerts: float
        the standard deviations of the parameters
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
    for td in range(num_delays):
        sbyield.append(np.sum(matdat[i_lo:i_hi, td]))

    params, uncerts = getPhase(sbyield, p0=[0, 2, 0, 0, 0], num_delays=num_delays)
    return params, uncerts

####################################
def energy_resolved_rabbitt_phase(momenta, matdat, sb, ip=0, fourp=False, fivep=False, 
                                  num_delays=16, intensity='1E12', manual_shift=0,
                                  photon_energy=0.05875):
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
    params : 2D array of floats
        the fitted parameters extracted from the rabbitt data, for each energy grid point
    uncerts : 2D array of floats
        the uncertainties associated with the params, also per energy grid point
    interval_lims : the spectral region over which the params were extracted
    """
    energies = mom_to_energy(ip, momenta)
    if sb in (16,18):
        photon_energy -= 0.0001
    for ii, en in enumerate(energies):
        if en > sb*photon_energy-0.01:
            i_lo = ii
            break
    for ii, en in enumerate(energies):
        if en > sb*photon_energy+0.01:
            i_hi = ii
            break

    sb_width = 0.01
    pond_shift = get_pondermotive_shift_eV(intensity, photon_energy)/27.21138
    print(f'Accounting fo pondermotive shift of: {pond_shift:.4e}')
    sb_energy = sb*photon_energy - pond_shift + (manual_shift/27.21138)
    if not fivep:
        sb_lo = sb_energy - sb_width
        i_lo = np.argmin(np.abs(energies-sb_lo))
    else:
        sb_lo = sb_energy 
        i_lo = np.argmin(np.abs(energies-sb_lo))
    if not fourp: 
        sb_hi = sb_energy + sb_width
        i_hi = np.argmin(np.abs(energies-sb_hi))
    else:
        sb_hi = sb_energy
        i_hi = np.argmin(np.abs(energies-sb_hi))

    params, uncerts = [], []
    interval_lims = (i_lo,i_hi)

    for en_idx in range(i_lo,i_hi):
        sbyield = matdat[en_idx, :]
        ps, us = getPhase(sbyield, num_delays=num_delays)
        params.append(ps)
        uncerts.append(us)

    return np.array(params), np.array(uncerts), interval_lims

def rabbitt(args, limit=1000):
    """Read momentum distribution from file, integrate over angle, and plot the
    rabbitt spectrum"""

    fourp, fivep = False, False
    if args['res'] == 4:
        fourp = True
    elif args['res'] == 5:
        fivep = True

    fullpsi = pd.read_csv(args['file'], index_col=0)
    Psi_phi = np.transpose(fullpsi.values)
    Psi_phi = Psi_phi[:limit]
    momenta = [float(x) for x in fullpsi.columns[:limit]]
    energies = mom_to_energy(0.904, momenta)
    matdat = np.zeros((len(energies), args['num_delays']))
    for td in range(args['num_delays']):
        matdat[:, td] = integrateOverAngle(Psi_phi[:, td*360:(td+1)*360])
    if args['plot']:
        plot_rabbit(matdat, energies, num_delays=args['num_delays'])
    params, param_uncerts, lims = energy_resolved_rabbitt_phase(momenta, matdat, args['sb'], 
                                                                args['ip'], fourp, fivep, 
                                                                num_delays=args['num_delays'], 
                                                                intensity=args['intensity'],
                                                                manual_shift=args['manual_shift'],
                                                                photon_energy=args['energy'])
    print([energies[l] for l in lims])
    print([energies[l]-0.904 for l in lims])
    print([27.21138*(energies[l]-0.904) for l in lims])
    phs_2 = params[:,1]
    phs_uncert_2 = param_uncerts[:,1]
    phs_4 = params[:,3]
    phs_uncert_4 = param_uncerts[:,3]
    signal_ratio = np.mean(params[:,0]/params[:,2])
    """
    Fix unphysical phase jumps:
    """
    current_phase_2 = phs_2[0]
    current_phase_4 = phs_4[0]
    for i in range(1,len(phs_2)):
        if phs_2[i] - current_phase_2 < -np.pi:
            phs_2[i] += 2*np.pi
        elif phs_2[i] - current_phase_2 > np.pi:
            phs_2[i] -= 2*np.pi

        if phs_4[i] - current_phase_4 < -np.pi:
            phs_4[i] += 2*np.pi
        elif phs_4[i] - current_phase_4 > np.pi:
            phs_4[i] -= 2*np.pi

        current_phase_2 = phs_2[i]
        current_phase_4 = phs_4[i]

    plt.figure(figsize=(10,8))
    plt.plot(27.21138*energies[lims[0]:lims[1]], phs_2, alpha=0.666, label='2omega', color='magenta', lw=2.5)
    plt.fill_between(27.21138*energies[lims[0]:lims[1]],
                     phs_2 - phs_uncert_2,
                     phs_2 + phs_uncert_2,
                     alpha=0.25,
                     color='magenta')
    plt.plot(27.21138*energies[lims[0]:lims[1]], phs_4, alpha=0.666, label='4omega', color='indigo', lw=2.5)
    plt.fill_between(27.21138*energies[lims[0]:lims[1]],
                     phs_4 - phs_uncert_4,
                     phs_4 + phs_uncert_4,
                     alpha=0.25,
                     color='indigo')
    plt.legend()
    plt.ylim((-4,4))
    plt.ylabel('Phase', fontsize=18)
    plt.xlabel('Photon Energy (eV)', fontsize=18)
    plt.title(f"1-D 2- & 4-omega Phases, SB{args['sb']}\nSignal Ratio: {signal_ratio:.3f}", fontsize=18)
    filename = f"{args['label']}_{args['res']}_Phase_1D.png" if (args["label"] is not None) else f'{args["sb"]}_{args["res"]}_Phase_1D.png'
    plt.savefig(filename, dpi=300)
    plt.close()

####################################
def pwPhase(args):
    """read partial wave momentum distributions and calculate the absolute or
    relative phase associated with that partial wave"""
    pwPhase = []
    pwPhase_4omega = []
    tmp_phases = PADphase(args)
    pwPhase.append(tmp_phases[0])
    pwPhase_4omega.append(tmp_phases[1])
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
