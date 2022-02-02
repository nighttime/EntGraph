if [ "$#" -ne 1 ]; then
	echo "Extract and format event instances for select predicates with document timestamp"
	echo "Usage: extract_test_rels.sh <news_gen.json>"
	exit 1
fi

mkdir -p temporal_rels

# Step 1: identify relevant lines in the news_gen file using grep
# Step 2: extract relation strings using sed to reformat them into the output file

echo  "Binary extraction"
mkdir temporal_rels/binary

PREDS=('buy' 'purchase' 'own' 'sell' 'bid\.for' 'pay\.for' 'receive' 'acquire' 'marry' 'divorce' 'travel\.to' 'move\.to' 'fly\.to' 'at' 'be\.in' 'be\.at' 'arrive\.in' 'arrive\.at' 'run\.for' 'kill' 'murder' 'play\.for' 'score\.for' 'win\.against' 'compete\.against' 'pass' 'propose')
RELS=()
for PRED in ${PREDS[@]}; do
	PRED_PREFIX=$(echo $PRED | sed -r 's_(.*)\\.*_\1_')
	RELS+=("\($PRED_PREFIX\.1,$PRED\.2\)")
done

PREDS+=('elect\.2\.to\.2' 'marry\.2\.to\.2')
RELS+=('\(elect\.2,elect\.to\.2\)' '\(marry\.2,marry\.to\.2\)')

for i in "${!PREDS[@]}"; do
	PRED=${PREDS[$i]}
	REL=${RELS[$i]}
	echo $PRED ':' $REL
	# grep -E "\($REL::([^:]+::){2}(GE|EG)" $1 | sed -r 's_.*date":"([^"]*)".*\(('"$REL"'[^\}]*)\)"\}.*_\1	\2_g' > "temporal_rels/binary/$PRED.txt"
	grep -E "\($REL::([^:]+::){2}(GE|EG)" $1 | sed -r 's_.*date":"([^"]*)".*articleId":"([^"]*)".*\(('"$REL"'[^\}]*)\)"\}.*_\1	\3	\2_g' > "temporal_rels/binary/$PRED.txt"
	echo 'finished' $PRED
done

############
echo ""
echo "Unary extraction"
mkdir temporal_rels/unary

PREDS=('die' 'be\.candidate' 'be\.dead' 'be\.guilty' 'sentence')
RELS=()
for PRED in ${PREDS[@]}; do
	RELS+=("$PRED\.1")
done

PREDS_PASSIVE=('kill')
for PRED in ${PREDS_PASSIVE[@]}; do
	PREDS+=("$PRED\.2")
	RELS+=("$PRED\.2")
done

for i in "${!PREDS[@]}"; do
	PRED=${PREDS[$i]}
	REL=${RELS[$i]}
	echo $PRED ':' $REL
	# grep -E "\($REL::[^:]+::E" $1 | sed -r 's_.*date":"([^"]*)".*\(('"$REL"'[^\}]*)\)"\}.*_\1	\2_g' > "temporal_rels/unary/$PRED.txt"
	grep -E "\($REL::[^:]+::E" $1 | sed -r 's_.*date":"([^"]*)".*articleId":"([^"]*)".*\(('"$REL"'[^\}]*)\)"\}.*_\1	\3	\2_g' > "temporal_rels/unary/$PRED.txt"
	echo 'finished' $PRED
done

