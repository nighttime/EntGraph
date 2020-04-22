from typing import *

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
	for i, ps in enumerate(predicted):
		# ps = [v for v, score in ps]
		rank = min((ps.index(t) for t in true[i] if t in ps), default=-1) if ps else -2
		ranks.append(rank)
		score = 1 / min(rank+1, limit) if rank >= 0 else 0
		scores.append(score)
	data, bins = np.histogram(ranks, bins=[-2,-1,0,1,2,3,4,5,6,7,8,9,10,100])
	print('{} questions answered'.format(len([p for p in predicted if p])))
	print('{} containing correct answers'.format(np.sum(data[2:])))
	print(data)
	print(bins)
	return sum(scores) / len(scores)


def jaccard(x: Set[Any], y: Set[Any]) -> float:
	return len(x.intersection(y))/len(x.union(y))
