echo "generating causal graphs for multiple parameter settings..."

python metarel.py ../newscrawl_sims/newscrawl_modifiers_3_3/ ../eval_data/metarel_graphs/nc_BB_3_3_local_marg_30 --decision=margin --margin=0.30 --text-graphs
python entailment.py ../eval_data/metarel_graphs/nc_BB_3_3_local_marg_30/ ../eval_data/CGs/nc_BB_3_3_local_marg_30 --stage=local --keep-forward

python metarel.py ../newscrawl_sims/newscrawl_modifiers_3_3/ ../eval_data/metarel_graphs/nc_BB_3_3_local_marg_15 --decision=margin --margin=0.15 --text-graphs
python entailment.py ../eval_data/metarel_graphs/nc_BB_3_3_local_marg_15/ ../eval_data/CGs/nc_BB_3_3_local_marg_15 --stage=local --keep-forward

python metarel.py ../newscrawl_sims/newscrawl_modifiers_3_3/ ../eval_data/metarel_graphs/nc_BB_3_3_local_marg_08 --decision=margin --margin=0.08 --text-graphs
python entailment.py ../eval_data/metarel_graphs/nc_BB_3_3_local_marg_08/ ../eval_data/CGs/nc_BB_3_3_local_marg_08 --stage=local --keep-forward

echo "done"