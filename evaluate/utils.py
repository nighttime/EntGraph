import os
import datetime
import proposition
from proposition import Prop
import reference
import pickle
from collections import Counter
from entailment import prop_recognized_in_graphs
from answer_tf import format_prop_ppdb_lookup
from typing import *

def checkpoint():
	print('+ Checkpoint:', datetime.datetime.now().strftime('%H:%M:%S'))

def print_progress(progress, info='', bar_len=20):
	filled = int(progress*bar_len)
	print('\r[{}{}] {:.2f}% {}'.format('=' * filled, ' ' * (bar_len-filled), progress*100, info), end='')
	if filled == bar_len:
		print()

def analyze_questions(P_list, N_list, uu_graphs, bu_graphs, PPDB: Optional[Dict[str, Dict[str, float]]]=None):
	num_Qs = sum(len(qs) for qs in P_list + N_list)
	num_unary_Qs = len([q for qs in P_list + N_list for q in qs if len(q.args) == 1])
	num_binary_Qs = num_Qs - num_unary_Qs

	num_P_unary_Qs = len([q for qs in P_list for q in qs if len(q.args) == 1])
	num_P_binary_Qs = len([q for qs in P_list for q in qs if len(q.args) == 2])
	num_P_copula_Qs = len([q for qs in P_list for q in qs if len(q.args) == 1 and q.pred.startswith('be.')])
	num_N_copula_Qs = len([q for qs in N_list for q in qs if len(q.args) == 1 and q.pred.startswith('be.')])
	num_copula_Qs = num_P_copula_Qs + num_N_copula_Qs

	num_sets = len(P_list)

	pct_positive = (num_P_unary_Qs + num_P_binary_Qs) / num_Qs
	pct_positive_u = num_P_unary_Qs / num_unary_Qs if num_unary_Qs else 0
	pct_positive_b = num_P_binary_Qs / num_binary_Qs if num_binary_Qs else 0
	pct_unary = num_unary_Qs / num_Qs
	pct_copula_u = num_copula_Qs / num_unary_Qs if num_unary_Qs else 0
	pct_positive_copula = num_P_copula_Qs / num_copula_Qs if num_copula_Qs else 0
	print(
		'Generated {} questions ({:.1f}% u; {:.1f}% +) from {} sets: {} unary questions ({:.1f}% +) and {} binary questions ({:.1f}% +)'.format(
			num_Qs, pct_unary*100, pct_positive * 100, num_sets, num_unary_Qs, pct_positive_u * 100, num_binary_Qs,
			pct_positive_b * 100))
	# print('{} or {:.1f}% of unary questions are copulas ({:.1f}% +)'.format(num_copula_Qs, pct_copula_u * 100, pct_positive_copula * 100))

	# known_u_qs = [p for ps in P_list for p in ps if
	# 			  len(p.args) == 1 and uu_graphs and p.pred_desc() in uu_graphs[p.types[0]].nodes]
	# # known_b_qs = [p for ps in P_list for p in ps if
	# 			  # len(p.args) == 2 and bu_graphs and p.pred_desc() in bu_graphs['#'.join(p.basic_types)].nodes]
	# known_b_qs = [p for ps in P_list for p in ps if prop_recognized_in_graphs(p, bu_graphs)]

	known_u_qs = [p for ps in P_list + N_list for p in ps if
				  len(p.args) == 1 and (p.pred_desc() in uu_graphs[p.types[0]].nodes or any(p.pred_desc() in g.nodes for g in bu_graphs.values()))]
	known_b_qs = [p for ps in P_list + N_list for p in ps if
				  len(p.args) == 2 and p.pred_desc() in bu_graphs['#'.join(p.basic_types)].nodes]

	known_u_untyped_qs = [p for ps in P_list + N_list for p in ps if
						  len(p.args) == 1 and (prop_recognized_in_graphs(p, uu_graphs) or prop_recognized_in_graphs(p, bu_graphs))]
	known_b_untyped_qs = [p for ps in P_list + N_list for p in ps if
						  len(p.args) == 2 and prop_recognized_in_graphs(p, bu_graphs)]

	pct_known_u = len(known_u_qs) / num_unary_Qs if num_unary_Qs else 0
	pct_known_b = len(known_b_qs) / num_binary_Qs if num_binary_Qs else 0

	print('Questions recognized in typed graph nodes: {:.1f}% unary, {:.1f}% binary'.format(pct_known_u * 100,
																							   pct_known_b * 100))

	if PPDB:
		formatted_qs = [format_prop_ppdb_lookup(p) for ps in P_list  + N_list for p in ps]
		found_qs = [q for q in formatted_qs if q in PPDB]
		fallback_qs = [[' '.join(q.split()[:-i]) for i in range(1, len(q.split()))] + [q] for q in formatted_qs]
		found_fallback_qs = [1 for qs in fallback_qs if any(q in PPDB for q in qs)]

		pct_known_qs = len(found_qs)/num_Qs
		pct_known_fallback_qs = len(found_fallback_qs)/num_Qs

		print('Questions recognized in PPDB: {:.1f}%, with fallback: {:.1f}%'.format(pct_known_qs*100, pct_known_fallback_qs*100))
		print('PPDB graph stats: {} nodes, {} allnodes {} edges'.format(len(PPDB), len(set(PPDB.keys()) | set([k for vs in PPDB.values() for k,v in vs.items()])), sum([len(es) for es in PPDB.values()])))

	pct_known_untyped_u = len(known_u_untyped_qs) / num_unary_Qs if num_unary_Qs else 0
	pct_known_untyped_b = len(known_b_untyped_qs) / num_binary_Qs if num_binary_Qs else 0
	print('Questions recognized in untyped graph nodes: {:.1f}% unary, {:.1f}% binary'.format(pct_known_untyped_u * 100,
																									 pct_known_untyped_b * 100))
	# question_types = Counter([tuple(sorted(p.types)) for ps in P_list for p in ps if len(p.args) == 2])
	# big_types = ['person', 'organization', 'location', 'thing']
	# num_big_types = sum([count for types,count in question_types.items() if any(t in big_types for t in types)])
	# binary_big_type_share = num_big_types / num_binary_Qs if num_binary_Qs > 0 else 0
	# binary_small_type_share = 1 - binary_big_type_share if num_binary_Qs > 0 else 0
	# print('Binary big-type share: {:.2f}%, small-type share: {:.2f}%'.format(binary_big_type_share*100, binary_small_type_share*100))


def write_questions_to_file(P_list: List[List[Prop]], N_list: List[List[Prop]]):
	with open('../tf_questions_pos.txt', 'w+') as f:
		for ps in P_list:
			for p in ps:
				f.write(p.prop_desc())
				f.write('\n')
			f.write('\n')

	with open('../tf_questions_neg.txt', 'w+') as f:
		for ns in N_list:
			for n in ns:
				f.write(n.prop_desc())
				f.write('\n')
			f.write('\n')

	print('Questions written to file.')

def write_pred_cache_to_file(pred_cache: Dict[str, int]):
	with open('../top_preds.txt', 'w+') as f:
		for pred,count in pred_cache.items():
			term = proposition.extract_predicate_base_term(pred)
			f.write(term)
			f.write('\n')

def save_results_on_file(dest_folder: str,
						 Q_list: List[List[Prop]],
						 A_list: List[List[int]],
						 results: Dict[str, Tuple[List[List[float]], List[List[Dict[str, Prop]]]]]):
	# results_file = os.path.join(dest_folder, 'last_results.pkl')
	fname = reference.FINISH_TIME + '.pkl'
	results_file = os.path.join(dest_folder, 'saved_results', fname)
	data = {'questions': Q_list, 'answers': A_list, 'results': results}
	with open(results_file, 'wb+') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
		print('Results saved to {}'.format(results_file))

def read_results_on_file(source_file: str) -> Tuple[List[List[Prop]], List[List[int]], Dict[str, Tuple[List[List[float]], List[List[Dict[str, Prop]]]]]]:
	with open(source_file, 'rb') as f:
		data = pickle.load(f)
		Q_List, A_List, results = data['questions'], data['answers'], data['results']
		return Q_List, A_List, results

def save_props_on_file(props: List[Prop], dest_folder: str, fname: str):
	results_file = os.path.join(dest_folder, fname + '.pkl')
	with open(results_file, 'wb+') as f:
		pickle.dump(props, f, pickle.HIGHEST_PROTOCOL)
		print('Props saved to {}'.format(results_file))

def read_props_on_file(source_path: str) -> List[Prop]:
	with open(source_path, 'rb') as f:
		props = pickle.load(f)
		return props