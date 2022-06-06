import json
import os
import datetime
# import proposition
from proposition import Prop, format_prop_ppdb_lookup, extract_predicate_base_term
import reference
import pickle
from collections import Counter
import entailment
from typing import *

def checkpoint():
	print('+ Checkpoint:', datetime.datetime.now().strftime('%H:%M:%S'))

# For printing
BAR_LEN = 50
BAR = '=' * BAR_LEN
bar = '-' * BAR_LEN

def print_bar():
	print(bar)

def print_BAR():
	print(BAR)

def print_progress(progress, info='', bar_len=20):
	filled = int(progress*bar_len)
	try:
		term_width = os.get_terminal_size()[0]
	except:
		term_width = 80
	update = '\r[{}{}] {:.2f}% {}'.format('=' * filled, ' ' * (bar_len-filled), progress*100, info)
	padding = (term_width - (len(update) - 1)) * ' '
	print(update + padding, end='')
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

	return
	if uu_graphs and bu_graphs:
		known_u_qs = [p for ps in P_list + N_list for p in ps if
					  len(p.args) == 1 and (p.pred_desc() in uu_graphs[p.types[0]].nodes or any(p.pred_desc() in g.nodes for g in bu_graphs.values()))]
		known_b_qs = [p for ps in P_list + N_list for p in ps if
					  len(p.args) == 2 and p.pred_desc() in bu_graphs['#'.join(p.basic_types)].nodes]

		known_u_untyped_qs = [p for ps in P_list + N_list for p in ps if
							  len(p.args) == 1 and (entailment.prop_recognized_in_graphs(p, uu_graphs) or entailment.prop_recognized_in_graphs(p, bu_graphs))]
		known_b_untyped_qs = [p for ps in P_list + N_list for p in ps if
							  len(p.args) == 2 and entailment.prop_recognized_in_graphs(p, bu_graphs)]

		pct_known_u = len(known_u_qs) / num_unary_Qs if num_unary_Qs else 0
		pct_known_b = len(known_b_qs) / num_binary_Qs if num_binary_Qs else 0
		print('Questions recognized in typed graph nodes: {:.1f}% unary, {:.1f}% binary'.format(pct_known_u * 100,
																							   pct_known_b * 100))
		pct_known_untyped_u = len(known_u_untyped_qs) / num_unary_Qs if num_unary_Qs else 0
		pct_known_untyped_b = len(known_b_untyped_qs) / num_binary_Qs if num_binary_Qs else 0
		print('Questions recognized in untyped graph nodes: {:.1f}% unary, {:.1f}% binary'.format(pct_known_untyped_u * 100, pct_known_untyped_b * 100))


	if PPDB:
		formatted_qs = [format_prop_ppdb_lookup(p) for ps in P_list  + N_list for p in ps]
		found_qs = [q for q in formatted_qs if q in PPDB]
		fallback_qs = [[' '.join(q.split()[:-i]) for i in range(1, len(q.split()))] + [q] for q in formatted_qs]
		found_fallback_qs = [1 for qs in fallback_qs if any(q in PPDB for q in qs)]

		pct_known_qs = len(found_qs)/num_Qs
		pct_known_fallback_qs = len(found_fallback_qs)/num_Qs

		print('Questions recognized in PPDB: {:.1f}%, with fallback: {:.1f}%'.format(pct_known_qs*100, pct_known_fallback_qs*100))
		print('PPDB graph stats: {} nodes, {} allnodes {} edges'.format(len(PPDB), len(set(PPDB.keys()) | set([k for vs in PPDB.values() for k,v in vs.items()])), sum([len(es) for es in PPDB.values()])))


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

def write_q_pred_freqs_to_file(freqs_pos: Dict[str, int], freqs_neg: Dict[str, int]):
	with open('../tf_q_pred_freqs_pos_1.4m_GG.txt', 'w+') as f:
		f.write(json.dumps(freqs_pos, indent=2))

	with open('../tf_q_pred_freqs_neg_1.4m_GG.txt', 'w+') as f:
		f.write(json.dumps(freqs_neg, indent=2))

def write_pred_cache_to_file(pred_cache: Dict[str, int]):
	with open('../top_preds.txt', 'w+') as f:
		for pred,count in pred_cache.items():
			term = extract_predicate_base_term(pred)
			f.write(term)
			f.write('\n')

def save_results_on_file(dest_folder: str,
						 Q_list: Union[List[List[Prop]], List[Tuple[Prop, Prop]]],
						 A_list: Union[List[List[int]], List[int]],
						 results: Dict[str, Tuple[List[List[float]], List[List[Dict[str, Prop]]]]],
						 memo: Optional[str] = None,
						 name: Optional[str] = None):
	# results_file = os.path.join(dest_folder, 'last_results.pkl')
	fname = reference.FINISH_TIME + '.pkl'
	results_file = os.path.join(dest_folder, 'saved_results', fname)
	data = {'questions': Q_list, 'answers': A_list, 'results': results, 'memo': memo, 'name': name}
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

# (treatment.for.1,treatment.for.2) vancomycin::medicine infection::disease
def read_lh_prop(prop_str: str) -> Prop:
	bare_pred, typed_arg1, typed_arg2 = prop_str.split()
	types = [a.split('::')[1] for a in [typed_arg1, typed_arg2]]
	if types[0] == types[1]:
		types = [types[0] + '_1', types[1] + '_2']
		typed_arg1, typed_arg2 = typed_arg1 + '_1', typed_arg2 + '_2'
	pred_desc = '#'.join([bare_pred, *types])
	arg_desc = [a.replace('::', '#') for a in [typed_arg1, typed_arg2]]
	prop = Prop.from_descriptions(pred_desc, arg_desc)
	prop.set_entity_types('EE')
	return prop

def read_lh(path: str, directional=False) -> Tuple[List[Optional[Tuple[Prop, Prop]]], List[int], List[int]]:
	print('Reading {}'.format(path))
	Ent_list, A_list = [], []
	issue_ct = 0
	with open(path) as file:
		for line in file:
			parts = line.strip().split('\t')
			answer = 1 if parts[-1] == 'True' else 0

			if len(parts) != 3 or not all(parts):
				issue_ct += 1
				Ent_list.append(None)
				A_list.append(answer)
				continue

			rel_str1, rel_str2, _ = parts
			prop1, prop2 = read_lh_prop(rel_str1), read_lh_prop(rel_str2)
			Ent_list.append((prop2, prop1))
			A_list.append(answer)

	q_idx = list(range(len(Ent_list)))
	if directional:
		all_qa = {Ent_list[i]:A_list[i] for i in range(len(Ent_list))}
		dir_qs = []
		reverses = set()
		q_idx = []
		for i,q in enumerate(Ent_list):
			if not q:
				continue
			if q in reverses:
				q_idx.append(i)
				dir_qs.append((q, all_qa[q]))
				continue
			rev_q = tuple(reversed(q))
			if rev_q in all_qa and all_qa[q] != all_qa[rev_q]:
				q_idx.append(i)
				dir_qs.append((q, all_qa[q]))
				reverses.add(rev_q)
		new_Ent_list, new_A_list = tuple(zip(*dir_qs))
		Ent_list, A_list = new_Ent_list, new_A_list

	assert len(q_idx) == len(Ent_list)
	assert len(Ent_list) == len(A_list)

	# Teddy gets 1784
	return Ent_list, A_list, q_idx

def read_dataset(dataset: str, data_folder: str, test=False, directional=False) -> Tuple[List[Tuple[Prop, Prop]], List[int], List[int]]:
	if dataset == 'levy_holt':
		# fname = '{}_rels.txt'.format('test' if test else 'dev')
		# path = os.path.join(data_folder, 'datasets', 'levy_holt', fname)
		# return read_lh(path, directional=directional)
		if directional:
			fname = f'{"test" if test else "dev"}_dir_rels_v2.txt'
		else:
			fname = '{}_rels.txt'.format('test' if test else 'dev')
		path = os.path.join(data_folder, 'datasets', 'levy_holt', fname)
		return read_lh(path)
	elif dataset == 'sl_ant':
		fname = 'ant_{}_rels_1best.txt'.format('directional' if directional else 'full')
		path = os.path.join(data_folder, 'datasets', 'sl_ant_parsed', fname)
		return read_lh(path)