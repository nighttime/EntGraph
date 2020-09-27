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