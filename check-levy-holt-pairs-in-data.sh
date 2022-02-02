if [ "$#" -ne 1 ]; then
	echo "Check if each relation instance in Levy-Holt dev set occurs in a training corpus"
	echo "Usage: check-levy-holt-pairs-in-data.sh <news_gen.json>"
	exit 1
fi

# touch lh-check.txt
DATAFILE=$1

escape_chars () {
	echo $(echo $1 | sed -E 's_\)_\\)_' | sed -E 's_\(_\\(_' | sed -E 's_\._\\._g')
}

grep_input_file () {
	echo $(grep -m 1 -c -E "\($1::([^:]+::){2}(GE|EG|EE)" $2) $1
}

cat levy-holt/dev_rels.txt | while read line; do
	L=($line)
	
	first=$(escape_chars ${L[0]})
	second=$(escape_chars ${L[3]})
	# echo $first
	
	echo $(grep_input_file $first $DATAFILE)
	echo $(grep_input_file $second $DATAFILE)
	
done

echo "done"