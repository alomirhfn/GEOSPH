#!/bin/bash

# Run the pysph application
OUTDIR="$3"
DIR=/data/Favero_Group/SD_PySPH/

cd ${DIR} || echo "Directory not found!"
opt=""

# PySPH options
g="0.0"  # Must set to zero in simulation where there is no gravity
tf="1.0"
dt="0.00005"
kh="1.3"
simdim=2
kgc=1 # kernel gradient correction: 0 - Off, 1 - On
intgrtr=1  # Tipe of integrator: 0 - Forward Euler, 1 - Leap-Frog, 2 - PC, 3 - Verlet, 4 - Fluid
fq=400 # Output dumping frequency
ssrfq=0  # Stress and strain regularization frequency
dbg=0
dbbc=0
alpha="0.4"  # Artificial bulk viscosity alpha_pi (< 1.0)
beta="0.1"  # Artificial shear viscosity alpha_mu (< 1.0)
c_model=1  # 0 - Elastic, 1 - Elastoplastic
yc=2  # 1 - Von Mises, 2 - Drucker-Prager, 3 - MCC, 4 - Mohr-Coulomb, 5 - CASM
d_time="0.0"  # Duration of kinematic dampening
calcn=0  # Calculate normals or use them from file, 1 - Calculate, 0 - File
sim_type=1  # 1 - single phase, 2 - undrained
drn=0  # Drained behavior of free surfaces (1 for drained)
sigma_c="0.0"  # Confining stress (- for compression)
bulkw="1e7"  # Elastic bulk modulus of water
dampeps="0.01"  # Zero or very low value for non-initialization simulation
#W="CubicSpline"
W="WendlandQuintic"
#opt="--max-s=1"

if [[ $# -gt 1 ]]; then
  {
    if [[ $2 = "omp" ]]; then
      par="--openmp"
    elif [[ $2 = "mpi" ]]; then
      par="mpi"
    elif [[ $2 = "no-omp" ]]; then
      par="--no-openmp"
    else
      par="?"  # Invalid option
    fi
  }
fi

if [[ ${par} = "mpi" ]]; then # Run with MPI
  {
    mpirun -n $1 python sd_pysph.py -d="${OUTDIR[*]}" --gravity=${g} \
    --tf=${tf} --timestep=${dt} --kernel-grad-correct=${kgc} --pfreq=${fq} \
    --monaghan-alpha=${alpha} --debug=${dbg} --c_model=${c_model} \
    --y_criterion=${yc} --damp-time=${d_time} --monaghan-beta=${beta} \
    --kernel=${W} --kh=${kh} --simdim=${simdim} \
    --sim-type=${sim_type} --debug-bound=${dbbc} --kdamp-eps=${dampeps} \
    --drained-fs=${drn} --bulkmod-w=${bulkw} --calc-normals=${calcn} \
    --sigma_c=${sigma_c} --integrator=${intgrtr} --ssr_freq=${ssrfq} ${opt}
  }
elif [[ ${par} != "mpi" ]]; then  # Run with/without OpenMP
  {
    if [[ ${par} = "--openmp" || ${par} = "--no-openmp" ]]; then
      {
        export OMP_NUM_THREADS=$1
        python sd_pysph.py -d="${OUTDIR[*]}" --gravity=${g} --tf=${tf} \
        --timestep=${dt} --kernel-grad-correct=${kgc} --pfreq=${fq} \
        --monaghan-alpha=${alpha} --debug=${dbg} --c_model=${c_model} \
        --y_criterion=${yc} --damp-time=${d_time} --monaghan-beta=${beta} \
        --kernel=${W} --kh=${kh} --simdim=${simdim} \
        --sim-type=${sim_type} --debug-bound=${dbbc} --kdamp-eps=${dampeps} \
        --drained-fs=${drn} --bulkmod-w=${bulkw} --calc-normals=${calcn} \
        --sigma_c=${sigma_c} --integrator=${intgrtr} --ssr_freq=${ssrfq} ${opt}
      }
    else
      {
        echo "#########################################"
        echo " Error: Invalid parallel paradigm option."
        echo " Valid arguments are 'omp', 'no-omp' and "
        echo " 'mpi'."
        echo "#########################################"
        echo
      }
    fi
  }
fi
