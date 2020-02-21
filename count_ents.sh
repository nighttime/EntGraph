fname="government#location_sim.txt"
echo $fname

total_lines=$(wc -l < $fname)
echo "total lines: $total_lines"

total_sub_preds=$(egrep -c "predicate: \[sub" $fname)
total_obj_preds=$(egrep -c "predicate: \[obj" $fname)
total_una_preds=$(egrep -c "predicate: \[una" $fname)

echo "total sub: $total_sub_preds"
echo "total obj: $total_obj_preds"
echo "total unary: $total_una_preds"



