import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Argon
# Ip = 0.579 # in a.u.
# Neon
#Ip = 0.79245
# sb = 14    # sideband number


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
    parser.add_argument('files', nargs="+",
                        help="list of OuterWave_momentum files ")
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
    parser.add_argument('-r', '--rskip', type=int,
                        help="rskip parameter", default=200)
    parser.add_argument('-i', '--ip', type=float,
                        help="ionisation energy in a.u. (default He)",
                        default=0.903569)
    parser.add_argument('-s', '--sb', type=int,
                        help="sideband index",
                        default=18)

    return vars(parser.parse_args())


args = read_command_line()
Ip = args['ip']
sb = args['sb']

photon_energy = 0.06          # in a.u.
sb_width = 0.01              # sideband will be summed over the range sb_energy ± sb_width

# PARAMETERS from RMT
rskip = args["rskip"]
del_r = 0.08              # grid spacing
# number of outer region points minus the first 200 a.u worth of points (200 is the default in reform)
Nt = 64488-int(rskip/del_r)
# number of momentum values per angle in the OuterWave_momentum.* files
Nr = 800

#########################################################
p0 = [0, 2, 0, 0]
lb = -1.5*np.pi
ub = 1.5*np.pi

sb_energy = sb*photon_energy  # energy of the sideband in a.u
# use Nt ie total no.of points to get factor=dk
factor = (2.0*np.pi)/(del_r*Nt)
zeniths = np.zeros(Nr)
for l in range(0, Nr):
    zeniths[l] = Ip+0.5*l*l*factor*factor  # convert from momentum to energy

# Build the list of indices which lie within the energy range of the sideband
sb_indices = []
for jj in range(len(zeniths)):
    zz = zeniths[jj]
    if np.abs(sb_energy-zz) < sb_width:
        sb_indices.append(jj)


def test_func(x, a, b, c, d):
    return a * np.cos(b * x + c) + d


def getPhase(data, angle=0, maxyield=10000):
    """
    fit the data and extract the phase
    """
    global p0, lb, ub
    rat = sum(data) / maxyield
#    print (sum(data), maxyield, sum(data) < 0.01*maxyield)
#    if sum(data) < 0.01*maxyield:
#        return -3.0
#    data *= 1e8
    phase_delays = [i*np.pi/8 for i in range(16)]
    params, params_covariance = curve_fit(test_func, np.array(phase_delays),
                                          data, p0=p0,
                                          bounds=(
                                              [-np.inf, 1.99, lb, -np.inf], [np.inf, 2.01, ub, np.inf]),
                                          maxfev=1e8, ftol=1e-14)
    p0 = [x for x in params]

#    if angle == 0:
#        plt.plot(phase_delays, test_func(np.array(phase_delays), *params), 'r',
#                 label="0")
#        plt.plot(phase_delays, data,'ro')
#    if angle > 0 and angle < 20:
#        plt.plot(phase_delays, test_func(np.array(phase_delays), *params), 'b',
#                 label="fit")
#        plt.plot(phase_delays, data,'bo', label="data")
#        plt.title(f"Emission angle: {angle}$^{{\circ}}$")
#        plt.legend()
#        plt.savefig(f"{angle}.png")
#        plt.close()

    return rat, params[2]


def get_yield(data, sb_indices, zeniths):
    """
    sum the contribution of data[ii] for ii in sb_indices
    """
    nsum = 0
    fac = [1 for _ in sb_indices]
    fac[0] = 0.5
    fac[-1] = 0.5
    for jj, ii in enumerate(sb_indices):
        nsum += fac[jj]*data[ii]*0.5*(zeniths[ii-1]+zeniths[ii+1])
    return (nsum)


def extract_phase(flist, refangle):
    """
    load the momentum distribution, then for each angle extract the phase. 
    """

    if [len(flist) == 1]:  # if data has been preformatted into a single file
        Psi_phi = np.loadtxt(flist[0])
    else:  # otherwise data is housed in separate files
        Psi_phi = np.array([])
        for fname in flist:
            tmp_phi = np.loadtxt(fname)
            # remove the last, repeated row (360° == 0°)
            Psi_phi = np.append(Psi_phi, tmp_phi[:-1])
    Psi_phi = np.reshape(Psi_phi, (16*360, Nr))
    Psi = np.transpose(Psi_phi)

    plt.plot(zeniths, Psi[:, 0], 'r')
    for ii in sb_indices:
        plt.plot(zeniths[ii], Psi[ii, 0], 'b.')

    plt.show()
    plt.close()
    yield14 = []
    for ii in range(16):
        tempyield = []
        tempsi = Psi[:, ii*360:(ii+1)*360]
        for jj in range(360):
            tempyield.append(get_yield(tempsi[:, jj], sb_indices, zeniths))

        yield14.append(tempyield)

    y14 = np.array(yield14)
    maxyield = 0
    for ii in range(360):
        maxyield = max(maxyield, sum(y14[:, ii]))

    phase = []
    ratio = []

    for ii in range(0, 360):
        rat, phs = getPhase(y14[:, ii], angle=ii, maxyield=maxyield)
        phase.append(1/np.pi*phs)
        ratio.append(rat)

    rat, phs = getPhase(y14[:, 0], angle=0, maxyield=maxyield)
    phase[0] = (1/np.pi*phs)
    ratio[0] = rat
    angles = np.linspace(0, 2*np.pi, 360)
    refphase = phase[refangle]
#    refphase = 0
    phase = [p-refphase for p in phase]
    return angles, phase, ratio


args = read_command_line()
x, phi, ratio = extract_phase(args["files"], args["angle"])
rat = [ii/max(ratio) for ii in ratio]
ratio = [a for a in rat]
np.savetxt(args["output"], np.column_stack((x, phi)))

if args["polar"]:
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for ang, phs, rat in zip(x, phi, ratio):
        ax.plot(ang, phs, '.', color=lighten_color('b', 2*rat))
    ax.set_theta_zero_location("S")
    ax.set_ylim([-1.2, 0.4])
    plt.title(f'{args["angle"]}')
    plt.savefig(f'{args["angle"]}')
    plt.show()
elif args["plot"]:
    plt.plot(x, phi)
    plt.xlabel("θ(°)")
    plt.ylabel("phase (π radians)")
    plt.show()
