import numpy as np
from sklearn import metrics
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import datetime
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

def threshold_sets(predictions: List[List[float]], thresh: float) -> List[List[int]]:
	return [threshold(ps, thresh) for ps in predictions]

def threshold(predictions: List[float], thresh: float) -> List[int]:
	return [1 if p > thresh else 0 for p in predictions]

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

def precision(true: List[int], predicted: List[int]) -> float:
	tp, fp, tn, fn = answer_distribution(true, predicted)
	return tp / (tp + fp)

def recall(true: List[int], predicted: List[int]) -> float:
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


def plot_results(folder: str, true: List[List[int]], prediction_results: Dict[str, List[List[float]]]):
	plt.figure(figsize=(9,7))

	# Plot random-guessing baseline
	naive_base = prediction_results['*always-true']
	# plt.hlines(naive_base, 0, 1, colors='red', linestyles='--', label='Always True')

	# Plot graph lines
	for title, predictions in prediction_results.items():
		if title.startswith('*'):
			continue
		true_flat, preds_flat = flatten_answers(true, predictions)
		precision, recall, thresholds = metrics.precision_recall_curve(true_flat, preds_flat)
		sns.lineplot(x=recall[1:], y=precision[1:], label=title)

	# Plot exact-match baseline
	true_flat, preds_flat = flatten_answers(true, prediction_results['*exact-match'])
	precision, recall, thresholds = metrics.precision_recall_curve(true_flat, preds_flat)
	sns.lineplot(x=[recall[1]], y=[precision[1]], marker='D', markersize=8, label='Exact-Match Only')

	plt.legend()
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('True-False Question Performance for Multivalent EGs (Person Filtered)')

	now = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')
	fname = folder + ('' if folder.endswith('/') else '/') + 'tf_results/' + now + '.png'
	plt.savefig(fname)
	print('Results figure saved to', fname)
	# plt.show()