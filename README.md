# rabbitt utilities

Utilities for working with photoelectron angular momentum distributions from
RMT-calculations and experiment as part of the "Atomic partial wave meter by
attosecond coincidence metrology" paper by W. Jiang et al. Nat. Comms. 2022.

All code written by Andrew C. Brown 2022.

For certain functions, the ionisation potential in a.u needs to be provided as
an input argument (--ip). For others, the sideband index needs to be specified.
The parameters for the figures in the paper are:
For Argon: Ip = 0.579 a.u. sb = 14
For Neon: Ip = 0.793 a.u. sb = 18
For Helium: Ip = 0.904 a.u. sb =18

requirements can be installed with pip:

      pip install -r <path to rabbitt_utilities>/requirements.txt

utilities can be invoked as individual scripts to python:

      python <path to rabbitt_utilities>/src/PADphase.py <options> PAD_file

or the package can be installed with pip which installs utilities directly

      pip install <path to rabbitt_utilities>
      PADphase.py <options> PAD_file
