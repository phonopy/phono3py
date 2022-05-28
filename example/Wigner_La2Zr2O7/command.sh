#!/bin/bash
phono3py --nac --cell POSCAR  --fc2 --dim="2 2 2" --dim-fc2="4 4 4" --mesh="19 19 19" --tmin=300 --tmax=1000 --tstep=700 --sym-fc --isotope --br --read-gamma --wigner > tc_La2Zr2O7.out
