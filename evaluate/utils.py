import os
import datetime
import proposition
from proposition import Prop
import reference
import pickle
from typing import *

def checkpoint():
	print('+ Checkpoint:', datetime.datetime.now().strftime('%H:%M:%S'))

def print_progress(progress, info='', bar_len=20):
	filled = int(progress*bar_len)
	print('\r[{}{}] {:.2f}% {}'.format('=' * filled, ' ' * (bar_len-filled), progress*100, info), end='')
	if filled == bar_len:
		print()

def analyze_questions(P_list, N_list, uu_graphs, bu_graphs):
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

	known_u_qs = [p for ps in P_list for p in ps if
				  len(p.args) == 1 and uu_graphs and p.pred_desc() in uu_graphs[p.types[0]].nodes]
	known_b_qs = [p for ps in P_list for p in ps if
				  len(p.args) == 2 and bu_graphs and p.pred_desc() in bu_graphs['#'.join(p.basic_types)].nodes]

	pct_known_u = len(known_u_qs) / num_P_unary_Qs if num_P_unary_Qs else 0
	pct_known_b = len(known_b_qs) / num_P_binary_Qs if num_P_binary_Qs else 0

	print('Positive questions recognized in graph nodes: {:.1f}% unary, {:.1f}% binary'.format(pct_known_u * 100,
																							   pct_known_b * 100))

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
	results_file = os.path.join(dest_folder, 'last_results.pkl')
	data = {'questions': Q_list, 'answers': A_list, 'results': results}
	with open(results_file, 'wb+') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
		print('Results saved to {}'.format(results_file))

def read_results_on_file(source_folder: str) -> Tuple[List[List[Prop]], List[List[int]], Dict[str, Tuple[List[List[float]], List[List[Dict[str, Prop]]]]]]:
	results_file = os.path.join(source_folder, 'last_results.pkl')
	with open(results_file, 'rb') as f:
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