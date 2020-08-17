import datetime
from proposition import Prop
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
			f.write(pred)
			f.write('\n')