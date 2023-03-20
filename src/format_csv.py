"""utility to wrangle RMT OuterWave momentum files into the correct csv format

The OuterWave momentum files by default come out as space-delimited (i.e.
floating point numbers separated by four spaces) and with no header
information.

What we want is a comma separated variable (csv) format, and a header-row
which contains the momentum values.
"""
import pandas as pd


def read_command_line():
    from argparse import ArgumentParser as AP
    parser = AP()
    parser.add_argument('file',
                        help="OuterWave_momentum file")
    parser.add_argument('--dk',
                        help="""momentum spacing in a.u. required unless --npt
                        is set""",
                        type=float,
                        default=None)
    parser.add_argument('--npt',
                        help="""number of grid points used required unless
                        --dk is set""",
                        type=int,
                        default=None)
    parser.add_argument('--dR',
                        help="radial grid spacing in a.u.",
                        type=float,
                        default=0.08)
    parser.add_argument('--output',
                        help='Name of file for output',
                        default="output.csv")

    args = vars(parser.parse_args())
    if not args['dk'] and not args['npt']:
        raise RuntimeError("""you must provide either the momentum spacing in
        a.u. or the number of radial grid points used to compute the momentum
        distribution""")

    if args['npt'] and not args['dk']:
        from numpy import pi
        args['dk'] = 2.0 * pi / (args['npt'] * args['dR'])

    return args


def main():
    args = read_command_line()

    dk = args['dk']
    file = args['file']

    with open(file, 'r') as f:
        lines = f.readlines()

    if ("," in lines[1]) and (len(lines) == 5761):
        raise RuntimeError("File appears to be in correct format")

    num_columns = len(lines[0].split())

    momentum_values = [ii * dk for ii in range(num_columns)]

    df = pd.read_csv(file, delim_whitespace=True, header=None,
                     names=momentum_values)

    df.to_csv(args['output'], index=False)


if __name__ == '__main__':
    main()
