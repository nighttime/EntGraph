import os
import datetime
import statistics

import numpy as np
from sklearn import metrics
import seaborn as sns; sns.set()

from analyze import subsample_prt

sns.set()
import matplotlib.pyplot as plt
from proposition import *
import reference
from typing import *

def make_results_folder(folder: str, test=False, direct=False):
	# Make results directory
	now = reference.FINISH_TIME or datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')
	location = 'direct' if direct else ('local' if reference.RUNNING_LOCAL else 'server')
	eval_set = 'test' if test else 'dev'
	run_fname = now + '_' + location + '_' + eval_set
	run_folder = os.path.join(folder, run_fname)
	os.makedirs(run_folder)
	print('* Generated folder for analysis output: {}'.format(run_folder))
	return run_folder

def analyze_results(folder: str, Ent_list: List[Tuple[Prop, Prop]], A_list: List[List[int]], results: Dict[str, Tuple[List[List[float]], Any]], dataset: str, test=False, direct=False, directional=False):
	if direct:
		run_folder = make_results_folder(folder, test=test, direct=direct)
	else:
		run_folder = folder
	plot_results(run_folder, Ent_list, A_list, results, dataset, directional=directional)

def plot_results(folder: str, Ent_list, A_list: List[List[int]], results: Dict[str, Tuple[List[List[float]], Any]], dataset, directional=False):
	comp_types = ['BB', 'UU', 'BU', 'BB-LM']

	plt.figure(figsize=(8, 5))
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.xlim(0.0, 1.0)
	plt.ylim(0.0, 1.05)
	ax = plt.gca()
	ax.set_facecolor((0.95, 0.95, 0.95))

	num_steps = 500
	markersize = 7
	marker_color = '#444444'

	# Print chance baseline
	true_flat = [ans for anss in A_list for ans in anss]
	y_chance = statistics.mean(true_flat)
	print('Random Chance baseline: {}'.format(y_chance))
	# sns.lineplot(y=y_chance, label='Chance', linestyle='--', color='#EEEEEE')
	ax.axhline(y=y_chance, color="grey", ls='--', linewidth=0.75, label='Chance')

	# Print exact-match baseline
	lemma_base_recall = 0
	for title, (predictions, _) in results.items():
		if not title.startswith('*'):
			continue
		base_b_preds_flat = [p for ps in results[title][0] for p in ps]
		precision_b, recall_b, _ = metrics.precision_recall_curve(true_flat, base_b_preds_flat)
		base_b_precision, base_b_recall = precision_b[1], recall_b[1]
		if title.startswith('*Lemma'):
			lemma_base_recall = base_b_recall
		print('{} recall: {:.4f}'.format(title, base_b_recall))
		print()

	# Plot component graph lines
	for title, (predictions, _) in results.items():
		if title not in comp_types:
			continue

		preds_flat = [p for ps in predictions for p in (ps if ps else [0])]
		precision, recall, threshold = metrics.precision_recall_curve(true_flat, preds_flat)
		precision, recall = precision[1:], recall[1:]
		precision, recall, threshold = subsample_prt(precision, recall, threshold, num_steps)
		sns.lineplot(x=recall, y=precision, label=title)
		auc = metrics.auc(recall, precision)
		print('{} AUC: {:.4f}'.format(title, auc))
		print('Max recall: {:.4f}'.format(max(recall)))
		print()

		if not directional:
			cl_prec, cl_rec = zip(*[(p,r) for p,r in zip(reversed(precision), reversed(recall)) if p > 0.5])
			if len(cl_rec) > 1:
				sns.lineplot(x=cl_rec, y=cl_prec, label=title + ' (P>.50)')
				auc = metrics.auc(cl_rec, cl_prec)
				print('{} AUC: {:.4f} for prec > 0.5'.format(title, auc))
				print('Max recall: {:.4f}'.format(max(cl_rec)))
				print()

				clcl_prec, clcl_rec = tuple(zip(*[(p, r) for p, r in zip(cl_prec, cl_rec) if r >= lemma_base_recall])) or ([], [])
				if len(clcl_rec) > 1:
					sns.lineplot(x=clcl_rec, y=clcl_prec, label=title + ' (P>.50 & R>Lem)')
					auc = metrics.auc(clcl_rec, clcl_prec)
					print('{} AUC: {:.4f} for prec > 0.5 and rec >= lemma'.format(title, auc))
					print('Max recall: {:.4f}'.format(max(clcl_rec)))
					print()

	# Plot exact-match baseline
	for title, (predictions, _) in results.items():
		if not title.startswith('*'):
			continue
		base_b_preds_flat = [p for ps in results[title][0] for p in ps]
		precision_b, recall_b, _ = metrics.precision_recall_curve(A_list, base_b_preds_flat)
		base_b_precision, base_b_recall = precision_b[1], recall_b[1]
		plt.plot([base_b_recall], [base_b_precision], marker='v', markersize=markersize, label=title)

	plt.xlabel('Recall', fontsize=17)
	plt.xticks(np.arange(0, 1.01, step=0.1), fontsize=14)

	plt.ylabel('Precision', fontsize=17)
	plt.yticks(np.arange(0, 1.05, step=0.1), fontsize=14)
	plt.legend(fontsize=13)

	ds_name = reference.DS_SHORT_NAME_TO_DISPLAY[dataset]
	plt.title(f'{ds_name} Performance')

	fname = 'plot.png'
	fpath_results = os.path.join(folder, fname)
	plt.savefig(fpath_results)
	print('Results figure saved to', folder)
	# plt.show()