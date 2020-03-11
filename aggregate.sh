# echo "Running EntailGraphFactoryAggregator with options:" "$@"

USAGE="USAGE: ./aggregate.sh <memory-cap> <num-partitions>"

if [ "$#" -ne 2 ]; then
	echo "${USAGE}"
	exit 1
fi

# Size of Java heap space specified in M(B) or G(B) e.g. 500M, 75G
MEM=$1
# Number of partitions stored in $2. We initialize the program with an offset into the partitions which is 0-based
NUM_PARTITIONS=$2
PARTITION_MAX=$(expr $NUM_PARTITIONS - 1)

for i in $(seq 0 "${PARTITION_MAX}"); do
	# echo $i
	java "-Xmx${MEM}" "-cp" "lib/*:out/production/EntGraph" "entailment.vector.EntailGraphFactoryAggregator" "--resume" "${NUM_PARTITIONS}" "${i}"
	if [ "$?" -ne 0 ]; then
		exit 1
	fi
done
