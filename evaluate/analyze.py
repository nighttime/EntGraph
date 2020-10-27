from proposition import *
import utils

import numpy as np
from sklearn import metrics
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict, OrderedDict
import os
import json
import random
import argparse
import sys

from typing import *



def flatten_answers(A_list: List[List[Any]], predictions_list: List[List[Any]]):
	A_list_flat = []
	Prediction_list_flat = []
	for i, partition_list in enumerate(A_list):
		assert len(partition_list) == len(predictions_list[i])
		for j, ans in enumerate(partition_list):
			A_list_flat.append(ans)
			Prediction_list_flat.append(predictions_list[i][j])

	assert len(A_list_flat) == len(Prediction_list_flat)

	return A_list_flat, Prediction_list_flat

def threshold_sets(predictions: List[List[float]], thresh: float, strict:bool=False) -> List[List[int]]:
	return [threshold(ps, thresh, strict=strict) for ps in predictions]

def threshold(predictions: List[float], thresh: float, strict:bool=False) -> List[int]:
	if strict:
		return [1 if p > thresh else 0 for p in predictions]
	else:
		return [1 if p >= thresh else 0 for p in predictions]

def acc(true: List[int], predicted: List[int]) -> float:
	assert len(true) == len(predicted)
	correct = sum(1 if true[i] == predicted[i] else 0 for i in range(len(true)))
	return correct/len(true)

def answer_distribution(true: List[int], predicted: List[int]) -> Tuple[int, int, int, int]:
	assert len(true) == len(predicted)
	tp, fp, tn, fn = 0, 0, 0, 0
	for t,p in zip(true, predicted):
		if t:
			if p:
				tp += 1
			else:
				fn += 1
		else:
			if p:
				fp += 1
			else:
				tn += 1
	return tp, fp, tn, fn

def calc_precision(true: List[int], predicted: List[int]) -> float:
	tp, fp, tn, fn = answer_distribution(true, predicted)
	return tp / (tp + fp)

def calc_recall(true: List[int], predicted: List[int]) -> float:
	tp, fp, tn, fn = answer_distribution(true, predicted)
	return tp / (tp + fn)

def ans(true: List[Set[str]], predicted: List[List[str]]) -> float:
	correct = []
	for i, true_a in enumerate(true):
		# pred_a = [v for v, score in predicted[i]]
		pred_a = predicted[i]
		# Give credit if any predicted answer holds true in Q
		if any(a in true_a for a in pred_a):
			correct.append(1)
		else:
			correct.append(0)
	return sum(correct) / len(correct)


def MRR(true: List[Set[str]], predicted: List[List[str]], limit: int=10) -> float:
	# predicted = [l[:limit] for l in predicted]
	ranks = []
	scores = []

	no_ans_no_pred = 0
	no_ans_pred_given = 0
	true_answer_total = 0
	pred_answer_total = 0

	for i, ps in enumerate(predicted):
		if not true[i]:
			if ps:
				no_ans_pred_given += 1
				pred_answer_total += len(ps)
			else:
				no_ans_no_pred += 1
			continue
		# ps = [v for v, score in ps]
		rank = min((ps.index(t) for t in true[i] if t in ps), default=-1) if ps else -2
		ranks.append(rank)
		score = 1 / min(rank+1, limit) if rank >= 0 else 0
		scores.append(score)
		pred_answer_total += len(ps)
		true_answer_total += len(true[i])

	unanswerable = no_ans_no_pred + no_ans_pred_given
	print('Unanswerable || {}/{:.1f}% correctly unanswered | {} answers given'.format(no_ans_no_pred, 100*no_ans_no_pred/unanswerable, no_ans_pred_given))
	data, bins = np.histogram(ranks, bins=[-2,-1,0,1,2,3,4,5,6,7,8,9,10,100])
	unanswered = data[0]
	answered = len(predicted) - unanswerable - unanswered
	TP = np.sum(data[2:])
	FP = data[1]
	print('Answerable   || Unanswered: {} | Answered: {}/{:.1f}% ({}/{:.1f}% contain TP | {} FP)'.format(unanswered, answered, 100*answered/(answered+unanswered), TP, 100*TP/(TP+FP), FP))
	print('Rank counts: {}'.format(str(dict(list(zip(bins[2:-2], data[2:-1]))+[(str(bins[-2])+'+', data[-1])]))))
	print('Average answers/question: true: {:.2f}, predicted: {:.2f}'.format(true_answer_total/len(true), pred_answer_total/(answered + no_ans_pred_given)))
	score = sum(scores) / len(scores)
	return score


def jaccard(x: Set[Any], y: Set[Any]) -> float:
	return len(x.intersection(y))/len(x.union(y))

def calc_adjusted_auc(precision, recall, recall_thresh):
	auc = metrics.auc(recall, precision)
	adjusted_auc = (auc - recall_thresh)
	normalized_adjusted_auc = adjusted_auc / (1 - recall_thresh)
	return adjusted_auc, normalized_adjusted_auc

# Take values corresponding to the mask if True and filter out if False
def filter_2D_data(data, mask):
	return [[data[i][j] for j in range(len(data[i])) if mask[i][j]] for i in range(len(data))]

def plot_results(folder: str,
				 Q_list: List[List[Prop]],
				 A_list: List[List[int]],
				 prediction_results: Dict[str, Tuple[List[List[float]], List[List[Dict[str, Prop]]]]],
				 sample: bool = False,
				 save_thresh: bool = False,
				 subset: Optional[str] = None):
	comp_types = ['BB', 'UU', 'BU']

	if subset is not None:
		arg_target = 1 if subset == 'unary' else 2
		mask = [[True if len(q.args) == arg_target else False for q in qs] for qs in Q_list]
		Q_list = filter_2D_data(Q_list, mask)
		A_list = filter_2D_data(A_list, mask)
		prediction_results = {title:(filter_2D_data(ps, mask), filter_2D_data(ss, mask)) for title,(ps,ss) in prediction_results.items()}

		skip = ['BB'] if subset == 'unary' else ['UU', 'BU']
		prediction_results = {title:preds for title,preds in prediction_results.items() if title not in skip}
		print('Analyzing with {} subset only...'.format(subset))

	plt.figure(figsize=(9,7))
	plt.ylim(0.0, 1.05)
	plt.xlim(0.0, 1.05)
	# plt.ylim(0.3, 1.05)
	# plt.xlim(0.2, 1.05)
	# plt.margins(0.1)

	# Plot random-guessing baseline
	# naive_base = prediction_results['*always-true']
	# plt.hlines(naive_base, 0, 1, colors='red', linestyles='--', label='Always True')

	# Save information for analysis
	test_precision = 0.8
	cutoffs = {}
	thresholds = {}
	precisions = {}
	recalls = {}

	# Calculate exact-match baseline
	# true_flat, em_preds_flat = flatten_answers(A_list, prediction_results['*exact-match U'][0])
	# precision, recall, threshold = metrics.precision_recall_curve(true_flat, em_preds_flat)
	# em_precision, em_recall = precision[1], recall[1]

	# Plot graph lines
	true_flat = [ans for anss in A_list for ans in anss]
	for title, (predictions, support) in prediction_results.items():
		if title.startswith('*'):
			continue

		preds_flat = [p for ps in predictions for p in ps]
		precision, recall, threshold = metrics.precision_recall_curve(true_flat, preds_flat)
		precision, recall = precision[1:], recall[1:]
		# sns.lineplot(x=recall[:-1], y=threshold[:-1], label=title)
		if title in comp_types:
			sns.lineplot(x=recall, y=precision, label=title, alpha=0.3)
		elif title in ['BERT', 'RoBERTa']:
			sns.lineplot(x=recall, y=precision, label=title)
		else:
			sns.lineplot(x=recall, y=precision, label=title)

		# adjusted_auc, normalized_adjusted_auc = calc_adjusted_auc(precision, recall, em_recall)
		# print('{} adjusted AUC = {:.3f} ({:.3f} normalized)'.format(title, adjusted_auc, normalized_adjusted_auc))

		prec_score_thresh_idx = np.abs(precision - test_precision).argmin()
		prec_score_thresh = threshold[prec_score_thresh_idx]
		cutoffs[title] = prec_score_thresh
		thresholds[title] = threshold
		precisions[title] = precision
		recalls[title] = recall

	# Generate unions
	comp_labels = [c for c in comp_types if c in prediction_results]
	# comp_labels, components = tuple(zip(*((k,v[0]) for k,v in prediction_results.items() if k in comp_types)))
	components = [prediction_results[label][0] for label in comp_labels]
	for ind in range(2, len(comp_labels)+1):
		num_steps = 1000
		union_labels = comp_labels[:ind]
		union_models = components[:ind]
		print('Generating union results for {} from {} sample points'.format(union_labels, num_steps))

		# Preallocate space for precision, recall
		optimistic_union = np.zeros([2, num_steps])

		for i,prec in enumerate(np.arange(1.0, 0, -1/num_steps)):
			threshes = [thresholds[l][np.abs(precisions[l] - prec).argmin()] for l in union_labels]
			threshed_classifications = np.array([[c for cs in threshold_sets(comp, thresh) for c in cs] for comp,thresh in zip(union_models, threshes)])
			optim_union_classifications: List[int] = np.any(threshed_classifications, axis=0).astype(int).tolist()

			optimistic_union[0][i] = calc_precision(true_flat, optim_union_classifications)
			optimistic_union[1][i] = calc_recall(true_flat, optim_union_classifications)

		# Cut off points before the dead drop to 1.0 recall
		try:
			opt_first_recall_1_ind = np.where(optimistic_union[1] == 1)[0][0]
			optimistic_union = optimistic_union[:,:opt_first_recall_1_ind]
		except:
			print('recall in union != 1 anywhere?')

		# Add initial point in line with the component models
		optimistic_union = np.insert(optimistic_union, 0, [1, 0], axis=1)

		sns.lineplot(x=optimistic_union[1], y=optimistic_union[0], label=' + '.join(union_labels))

		# opt_adjusted_auc, opt_normalized_adjusted_auc = calc_adjusted_auc(optimistic_union[0], optimistic_union[1], em_recall)
		# print('Optimistic Union adjusted AUC = {:.3f} ({:.3f} normalized)'.format(opt_adjusted_auc, opt_normalized_adjusted_auc))

	# Plot exact-match baseline
	if '*exact-match U' in prediction_results and subset != 'binary':
		true_flat, em_u_preds_flat = flatten_answers(A_list, prediction_results['*exact-match U'][0])
		precision_u, recall_u, _ = metrics.precision_recall_curve(true_flat, em_u_preds_flat)
		em_u_precision, em_u_recall = precision_u[1], recall_u[1]
		sns.lineplot(x=[em_u_recall], y=[em_u_precision], marker='D', markersize=5, label='Exact-Match U')

	if '*exact-match B' in prediction_results and subset != 'unary':
		true_flat, em_b_preds_flat = flatten_answers(A_list, prediction_results['*exact-match B'][0])
		precision_b, recall_b, _ = metrics.precision_recall_curve(true_flat, em_b_preds_flat)
		em_b_precision, em_b_recall = precision_b[1], recall_b[1]
		sns.lineplot(x=[em_b_recall], y=[em_b_precision], marker='D', markersize=5, label='Exact-Match B')

	if '*exact-match U' in prediction_results and '*exact-match B' in prediction_results and subset is None:
		true_flat, em_u_preds_flat = flatten_answers(A_list, prediction_results['*exact-match U'][0])
		true_flat, em_b_preds_flat = flatten_answers(A_list, prediction_results['*exact-match B'][0])
		em_ub_preds_flat = [1 if (u == 1 or b == 1) else 0 for u,b in zip(em_u_preds_flat, em_b_preds_flat)]
		precision_ub, recall_ub, _ = metrics.precision_recall_curve(true_flat, em_ub_preds_flat)
		em_ub_precision, em_ub_recall = precision_ub[1], recall_ub[1]
		sns.lineplot(x=[em_ub_recall], y=[em_ub_precision], marker='D', markersize=5, label='Exact-Match U+B')

	plt.legend()

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	if subset is not None:
		plt.title('True-False Question Performance ({} Only)'.format(subset.title()))
	else:
		plt.title('True-False Question Performance')
	# plt.ylabel('Threshold')
	# plt.title('Threshold Levels for P-R Curve')

	now = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')
	location = '_local' if reference.RUNNING_LOCAL else '_server'
	subset_name = '_' + subset if subset is not None else ''
	fname = 'tf_results/' + now + subset_name + location + '.png'
	fpath_results = os.path.join(folder, fname)
	plt.savefig(fpath_results)
	print('Results figure saved to', fpath_results)
	# plt.show()

	if save_thresh:
		fname = 'tf_results/{}-PREC{:.2f}_thresholds.pkl'.format(now, test_precision)
		fpath_thresh = os.path.join(folder, fname)
		ob = {'precisions': precisions, 'thresholds': thresholds}
		with open(fpath_thresh, 'wb+') as f:
			pickle.dump(ob, f, pickle.HIGHEST_PROTOCOL)
			print('Results thresholds saved to', fpath_thresh)

	# Analyze results for specified precision level
	if sample:
		uu_preds = prediction_results['U->U'][0]
		bu_preds = prediction_results['B->U'][0]
		# uu_bu_preds = prediction_results['U->U and B->U'][0]
		uu_bu_adj_preds = prediction_results['U->U and B->U (Adj)'][0]
		# sim_preds = prediction_results['Similarity'][0]

		uu_class = threshold_sets(uu_preds, cutoffs['U->U'])
		bu_class = threshold_sets(bu_preds, cutoffs['B->U'])
		# uu_bu_class = threshold_sets(uu_bu_preds, cutoffs['U->U and B->U'])
		uu_bu_adj_class = threshold_sets(uu_bu_adj_preds, cutoffs['U->U and B->U (Adj)'])
		# sim_class = threshold_sets(sim_preds, cutoffs['Similarity'])

		# uu_dist = answer_distribution(true_flat, [x for xs in uu_class for x in xs])
		# bu_dist = answer_distribution(true_flat, [x for xs in bu_class for x in xs])

		result_infos = []
		ans = 1
		total_disagreements = 0
		# correct_union = 0
		adj_correct_union = 0
		for i, qs in enumerate(Q_list):
			# save question if it meets criteria
			for j, q in enumerate(qs):
				# if sim_preds[i][j] > 0.99 and not prediction_results['*exact-match'][0][i][j]:
				# 	sim_support = prediction_results['Similarity'][1][i][j] or '-'
				# 	result_infos.append({
				# 			'question': str(q),
				# 			'answer': ans,
				# 			'sim': '{:.2f} ({})'.format(sim_preds[i][j], sim_class[i][j]),
				# 			'sim support': str(sim_support)
				# 	})

				agreement = uu_class[i][j] == bu_class[i][j]
				correct_union_adj = uu_bu_adj_class[i][j] == ans

				if A_list[i][j] == ans:
					if not agreement:
						total_disagreements += 1
						if correct_union_adj:
							adj_correct_union += 1
						# if uu_bu_class[i][j] == ans:
						# 	correct_union += 1

					uu_support = prediction_results['U->U'][1][i][j] or '-'
					bu_support = prediction_results['B->U'][1][i][j] or '-'
					result_infos.append({
							'question': str(q),
							'answer': ans,
							'agreement': str(agreement),
							'correct_union': str(uu_bu_adj_class[i][j] == ans),
							'UU': '{:.2f} ({})'.format(uu_preds[i][j], uu_class[i][j]),
							'BU': '{:.2f} ({})'.format(bu_preds[i][j], bu_class[i][j]),
							# 'UU+BU': '{:.2f} ({})'.format(uu_bu_preds[i][j], uu_bu_class[i][j]),
							'UU+BU (A)': '{:.2f} ({})'.format(uu_bu_adj_preds[i][j], uu_bu_adj_class[i][j]),
							'UU support': str(uu_support),
							'BU support': str(bu_support)
					})


		fname = 'tf_results/{}_sample-PREC{:.2f}_UU{:.2f}_BU{:.2f}_UU+BU{:.2f}.txt'.format(now, test_precision, cutoffs['U->U'], cutoffs['B->U'], cutoffs['U->U and B->U (Adj)'])
		fpath_analysis = os.path.join(folder, fname)

		num_uu_class_1 = len([c for cs in uu_class for c in cs if c == 1])
		num_uu_answered = len([p for ps in uu_preds for p in ps if p > 0])

		num_bu_class_1 = len([c for cs in bu_class for c in cs if c == 1])
		num_bu_answered = len([p for ps in bu_preds for p in ps if p > 0])

		# num_uu_bu_class_1 = len([c for cs in uu_bu_class for c in cs if c == 1])
		# num_uu_bu_answered = len([p for ps in uu_bu_preds for p in ps if p > 0])

		num_uu_bu_adj_class_1 = len([c for cs in uu_bu_adj_class for c in cs if c == 1])
		num_uu_bu_adj_answered = len([p for ps in uu_bu_adj_preds for p in ps if p > 0])

		num_questions = len([q for qs in Q_list for q in qs])

		with open(fpath_analysis, 'w+') as f:
			f.write('PREC{:.2f}, {} total questions\n'.format(test_precision, num_questions))
			f.write('\n')

			f.write('=== Model Confidence ===\n')
			f.write('{}/{} UU answers above {:.2f} cutoff\n'.format(num_uu_class_1, num_uu_answered, cutoffs['U->U']))
			f.write('{}/{} BU answers above {:.2f} cutoff\n'.format(num_bu_class_1, num_bu_answered, cutoffs['B->U']))
			# f.write('{}/{} UU+BU answers above {:.2f} cutoff\n'.format(num_uu_bu_class_1, num_uu_bu_answered, cutoffs['U->U and B->U']))
			f.write('{}/{} UU+BU (A) answers above {:.2f} cutoff\n'.format(num_uu_bu_adj_class_1, num_uu_bu_adj_answered, cutoffs['U->U and B->U (Adj)']))
			f.write('\n')
			# f.write('{}/{} ({:.2f}%) correct union taken in disagreement cases\n'.format(correct_union, total_disagreements, correct_union/total_disagreements*100))

			f.write('=== Model Coverage ===\n')
			for name, classifications in [('UU', uu_class), ('BU', bu_class), ('UU+BU (A)', uu_bu_adj_class)]:
				tp, fp, tn, fn = answer_distribution(*flatten_answers(A_list, classifications))
				f.write(name + ':\t{} tp\t{} fp\t{} fn\n'.format(tp, fp, fn))
			f.write('\n')
			f.write('{}/{} ({:.2f}%) (Adj) correct union taken in disagreement cases\n'.format(adj_correct_union, total_disagreements, adj_correct_union/total_disagreements * 100))
			f.write('\n')

			sample_size = 500
			f.write('=== Sample (n={}) ===\n'.format(sample_size))
			result_sample = random.sample(result_infos, sample_size)
			for s in result_sample:
				f.write(json.dumps(s, indent=2))
				f.write('\n\n')

		print('Results analysis saved to', fpath_analysis)



def run():
	# global ARGS
	ARGS = parser.parse_args()
	data_folder = ARGS.data_folder
	print('Reading in results for analysis...')
	Q_List, A_List, results = utils.read_results_on_file(data_folder)
	print('Results read for {}'.format(list(results.keys())))
	if ARGS.plot:
		print('Performing analysis...')
		plot_results(data_folder, Q_List, A_List, results)

	print('Done')
	return Q_List, A_List, results


parser = argparse.ArgumentParser(description='Analyze data left in the data folder')
parser.add_argument('data_folder', help='Path to data folder including last results')
parser.add_argument('--plot', action='store_true', help='Run plotting procedure')

if __name__ == '__main__':
	Q, A, results = run()