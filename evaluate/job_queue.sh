echo 'STARTING JOBS'

python run_dataset.py ../eval_data --dataset sl_ant --bu-graphs ../eval_data/EGs/javad_tacl_BB_3_3_ctforw_global_slots.pkl --graph-embs ../eval_data/emb_EGs/hosseini2018_roberta/ --model roberta --plot --save-results --backoff=both_nodes --directional --smoothing-K=3 --smooth-P --memo="augment MISSING P, NO Q, K=3; positional weighting, disjoint answering, Javad 2018 global graph, dir ANT"

echo '>>>>>>>>>>>>> done 1'

python run_dataset.py ../eval_data --dataset sl_ant --bu-graphs ../eval_data/EGs/javad_tacl_BB_3_3_ctforw_global_slots.pkl --graph-embs ../eval_data/emb_EGs/hosseini2018_roberta/ --model roberta --plot --save-results --backoff=both_nodes --directional --smoothing-K=2 --smooth-P --memo="augment MISSING P, NO Q, K=2; positional weighting, disjoint answering, Javad 2018 global graph, dir ANT"

echo '>>>>>>>>>>>>> done 2'

python run_dataset.py ../eval_data --dataset sl_ant --bu-graphs ../eval_data/EGs/javad_tacl_BB_3_3_ctforw_global_slots.pkl --graph-embs ../eval_data/emb_EGs/hosseini2018_roberta/ --model roberta --plot --save-results --backoff=both_nodes --directional --smoothing-K=1 --smooth-P --memo="augment MISSING P, NO Q, K=1; positional weighting, disjoint answering, Javad 2018 global graph, dir ANT"

echo '>>>>>>>>>>>>> done 3'

python run_dataset.py ../eval_data --dataset sl_ant --bu-graphs ../eval_data/EGs/javad_tacl_BB_3_3_ctforw_global_slots.pkl --graph-embs ../eval_data/emb_EGs/hosseini2018_roberta/ --model roberta --plot --save-results --backoff=both_nodes --directional --smoothing-K=3 --smooth-Q --memo="augment NO P, MISSING Q, K=3; positional weighting, disjoint answering, Javad 2018 global graph, dir ANT"

echo '>>>>>>>>>>>>> done 4'

python run_dataset.py ../eval_data --dataset sl_ant --bu-graphs ../eval_data/EGs/javad_tacl_BB_3_3_ctforw_global_slots.pkl --graph-embs ../eval_data/emb_EGs/hosseini2018_roberta/ --model roberta --plot --save-results --backoff=both_nodes --directional --smoothing-K=2 --smooth-Q --memo="augment NO P, MISSING Q, K=2; positional weighting, disjoint answering, Javad 2018 global graph, dir ANT"

echo '>>>>>>>>>>>>> done 5'

python run_dataset.py ../eval_data --dataset sl_ant --bu-graphs ../eval_data/EGs/javad_tacl_BB_3_3_ctforw_global_slots.pkl --graph-embs ../eval_data/emb_EGs/hosseini2018_roberta/ --model roberta --plot --save-results --backoff=both_nodes --directional --smoothing-K=1 --smooth-Q --memo="augment NO P, MISSING Q, K=1; positional weighting, disjoint answering, Javad 2018 global graph, dir ANT"

echo '>>>>>>>>>>>>> done 6'

echo 'DONE'