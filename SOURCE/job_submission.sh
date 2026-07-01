#!/bin/bash

###############################################################################
# Help                                                                        #
###############################################################################
Help()
{
   # Display Help
   echo
   echo "#####################################################################"
   echo "job_submission.sh"
   echo "#####################################################################"
   echo "This script is used to submit a single simulation job for PySPH."
   echo "It takes exactly seven arguments as described below."
   echo "#####################################################################"
   echo
   echo "Syntax: bash job_submission.sh p N n m j d c t [-h]"
   echo
   echo "Arguments:"
   echo "  p     Partition name: 'short', 'medium', 'long' or 'ahf009."
   echo "  N     Number of nodes."
   echo "  n     Number of cores."
   echo "  m     Memory per core: 'XM' or 'XG'."
   echo "  j     Job name: '<name>'."
   echo "  d     Parallel paradigm: 'omp' or 'mpi'."
   echo "  c     Path to output directory: '/home/ahf009/<...>'."
   echo "  t     Simulation time: HH:MM:SS."
   echo
   echo "Options:"
   echo "  -h     Show this help."
   echo
}

###############################################################################
# Check arguments                                                             #
###############################################################################
Check_Args()
{
  # Check the number of arguments is correct
  if [ "$1" -ne 8 ]; then
    echo
    echo "####################################################################"
    echo "ERROR: Number of arguments must be equal to eight (8)."
    echo "####################################################################"
    Help
    exit 1
  fi
}

###############################################################################
# Main program                                                                #
###############################################################################

###############################################################################
# Process the input options. Add options as needed.                           #
###############################################################################
# Get the options
while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

Check_Args $#  # Check for the right number of arguments

###############################################################################
# Call SLURM's batch with command line arguments                              #
###############################################################################
# https://stackoverflow.com/questions/27708656/pass-command-line-arguments-via-sbatch

echo
echo "########################################################################"
echo "Job submitted with the following configuration:"
echo "Job name - $5"
echo "Partition - $1"
echo "Number of CPUs - $2"
echo "Number of cores - $3"
echo "Memory per core - $4"
echo "Parallel paradigm - $6"
echo "Output directory - $7"
echo "Allocated time - $8"
echo "########################################################################"
echo

###############################################################################
# Create Output Directory                                                     #
###############################################################################
if [ ! -d "$7" ]  # Create directory only if it exists
then
  mkdir "$7"
else
  echo "Directory already exists! Using existing directory."
  echo
fi

sbatch <<EOT
#!/bin/bash

#SBATCH -p $1  # partition (queue)
#SBATCH -N $2  # number of nodes (leave at 1 unless using multi-node code)
#SBATCH -n $3  # number of cores
#SBATCH --mem-per-cpu=$4  # memory per core (format XM or YG)
#SBATCH --time=$8  # Simulation estimated time
#SBATCH --job-name="$5"  # job name
#SBATCH -o /data/Favero_Group/STDOUT_2/slurm.%N."$5".stdout.txt # STDOUT
#SBATCH -e /data/Favero_Group/STDOUT_2/slurm.%N."$5".stderr.txt # STDERR
#SBATCH --mail-user=ahf009@bucknell.edu # address to email
#SBATCH --mail-type=ALL # mail events (NONE, BEGIN, END, FAIL, ALL)

# ==========================================================
# SANITIZE ANY INHERITED ENVIRONMENT
# ==========================================================
# 1. Clear any inherited virtual environments
unset VIRTUAL_ENV
unset PYTHONPATH
unset PYTHONHOME

# 2. Clear any inherited SLURM modules
module purge
# ==========================================================

module load python/3.10-ahf009
bash pysph_run.sh $3 "$6" "$7"

exit 0
EOT
echo