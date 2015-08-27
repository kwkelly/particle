#!/bin/bash
MAX_QUEUE_DEFAULT=20
WAIT_TIME_DEFAULT=60
MAX_TIME_DEFAULT=01:00:00
DEPTHS_DEFAULT=4
M_DEFAULT=8
N_DEFAULT=4
NS_DEFAULT=1
ND_DEFAULT=20
RS_DEFAULT=1
RD_DEFAULT=10
RB_DEFAULT=10
NSCATTERS_DEFAULT=25


usage="$(basename "$0") [-h] [-?] [-t -d -m -n --nd --rd --ns --rs --rb] -- run job script

where:
-h  Show this help text
-?  Show this help text
-t  max run time (hh:mm:ss)
-m  multipole order
-d  depth
-n  number of mpi tasks
--nd  number of detectors
--rd  reduced number of detectors
--ns  number of sources
--rs  reduced number of sources
--rb  B reduced number
--nscatters Number of scatterers"

while getopts "h?t:q:d:m:n:-:" opt; do
	case "$opt" in
		h|\?)
		echo "$usage"
		exit 0
		;;
		-)
			case "${OPTARG}" in
				nd) NDS="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
				;;
				ns) NSS="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
				;;
				rs) RSS="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
				;;
				rd) RDS="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
				;;
				rb) RBS="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
				;;
				nscatters) NSCATTERS="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
				;;
			esac
			;;
		t) MAX_TIME=$OPTARG
		;;
		d) DEPTHS=$OPTARG
		;;
		m) MS=$OPTARG
		;;
		n) NS=$OPTARG
		;;
		rd) RDS=$OPTARG
		;;
		nd) NDS=$OPTARG
		;;
		ns) NSS=$OPTARG
		;;
		rs) RSS=$OPTARG
		;;
		rb) RBS=$OPTARG
		;;
	esac
done
shift $((OPTIND-1))

MAX_TIME=${MAX_TIME:-$(echo $MAX_TIME_DEFAULT)}
DEPTHS=${DEPTHS:-$(echo $DEPTHS_DEFAULT)}
NS=${NS:-$(echo $N_DEFAULT)}
MS=${MS:-$(echo $M_DEFAULT)}
QS=${QS:-$(echo $Q_DEFAULT)}
RDS=${RDS:-$(echo $RD_DEFAULT)}
NDS=${NDS:-$(echo $ND_DEFAULT)}
NSS=${NSS:-$(echo $NS_DEFAULT)}
RSS=${RSS:-$(echo $RS_DEFAULT)}
RBS=${RBS:-$(echo $RB_DEFAULT)}
NSCATTERS=${NSCATTERS:-$(echo $NSCATTERS_DEFAULT)}


for M in ${MS}
do
for N in ${NS}
do
for ND in ${NDS}
do
for RD in ${RDS}
do
for NS in ${NSS}
do
for RS in ${RSS}
do
for RB in ${RBS}
do
for DEPTH in ${DEPTHS}
do
for NSCATTER in ${NSCATTERS}
do
	while : ; do
		[[ $(squeue -u $USER | tail -n +1 | wc -l) -lt $MAX_QUEUE_DEFAULT ]] && break
		echo "Pausing until the queue empties enough to add a new one."
		sleep $WAIT_TIME_DEFAULT
	done
	JOBNAME=tests-new-$NSCATTER-$ND-$RD
cat <<-EOS | sbatch
	#!/bin/bash

	#SBATCH -J $JOBNAME
	#SBATCH -o ../results/$JOBNAME.out
	#SBATCH -n $(( 1*$N ))
	#SBATCH -N $N
	#SBATCH -p gpu
	#SBATCH -t ${MAX_TIME:-$(echo $MAX_TIME_DEFAULT)}
	##SBATCH --mail-user=keith@ices.utexas.edu
	##SBATCH --mail-type=begin
	##SBATCH --mail-type=end
	#SBATCH -A PADAS
	cd ~/projects/particle/build/
	ibrun ./faims -fmm_m $M -min_depth $DEPTH -max_depth $DEPTH -dir /work/02370/kwkelly/maverick/files/results/ -R_d $RD -R_s $RS -N_d $ND -N_s $NS -R_b $RB -N_scatter $NSCATTER

	exit 0
	EOS
done
done
done
done
done
done
done
done
done
