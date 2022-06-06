# This file needs to be run in an environment with special packages including pytorch_transformers
import argparse
import math
import sys
import traceback
import os
from collections import defaultdict, Counter, namedtuple, OrderedDict
import pickle
from itertools import chain
import numpy as np
import sklearn.metrics
import torch
from transformers import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree

import article
import proposition
import reference
import utils
from graph_encoder import GraphDeducer
from run_dataset import read_dataset

from typing import *


def load_graph_type(typing, graph_embs_dir) -> Optional[Tuple[Dict[int, str], np.ndarray, BallTree]]:
	basic_typing = typing
	reverse_typing = '#'.join(reversed(basic_typing.split('#')))
	valency = 2

	embs_path = os.path.join(graph_embs_dir, basic_typing + '.pkl')
	if not os.path.exists(embs_path):
		embs_path = os.path.join(graph_embs_dir, reverse_typing + '.pkl')
	if not os.path.exists(embs_path):
		return None
	with open(embs_path, 'rb') as f:
		idx, embs = pickle.load(f)

		nonzero_rows = ~(embs == 0).all(1)
		embs = embs[nonzero_rows]
		inv_idx = {v: k for k, v in idx.items() if nonzero_rows[v]}

		# Use mask to keep only correct-valency candidate replacements
		keep_inv_idx = {i: p for i, p in inv_idx.items() if p.count('#') == valency}
		keep_idx = sorted(list(keep_inv_idx.keys()))
		embs = embs[keep_idx, :]
		inv_idx = {new_idx: inv_idx[old_idx] for new_idx, old_idx in enumerate(keep_idx)}

		# preprocessing for cosine similarity (norming)
		# norms = np.expand_dims(np.linalg.norm(embs, axis=1), axis=1)
		# embs = embs / norms

		if embs.shape[0] == 0:
			return None

		assert not np.isnan(embs).any()
		tree = BallTree(embs)

		return inv_idx, embs, tree

def compute_hierarchy(typing, target_preds: List[str], deducer: GraphDeducer, inv_idx: Dict[int, str], graph_embs: np.ndarray, graph_tree: BallTree, k=10) -> \
		Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]]]:
	graph_preds = set(inv_idx.values())

	def _find_NNs(query_pred, query_emb: np.ndarray, search_tree: BallTree, idx_to_pred: Dict[int,str], k=k) -> List[Tuple[str, float]]:
		search_k = min(search_tree.data.shape[0], k)
		top_k_dists, top_k_inds = search_tree.query(query_emb, k=search_k)
		top_k_scores = (1 / (1 + (top_k_dists))).tolist()[0]
		top_k_preds = [[idx_to_pred[i] for i in row] for row in top_k_inds][0]
		return list(zip(top_k_preds, top_k_scores))

	# Compute embedding matrix for graph + target nodes from dataset
	target_preds = [t for t in target_preds if t not in graph_preds]
	all_embedded = np.zeros([graph_embs.shape[0] + len(target_preds), graph_embs.shape[1]])
	all_embedded[0:len(graph_embs)] = graph_embs
	for i,pred in enumerate(target_preds):
		idx = len(graph_embs) + i
		all_embedded[idx] = deducer.encode_preds([pred])[0]

	# Compute BallTree of new embedding matrix to use for searching
	all_tree = BallTree(all_embedded)
	all_idx = {i:p for i,p in enumerate(list(inv_idx.values()) + target_preds)}

	# Compute hypernyms and hyponyms
	hypernyms = defaultdict(list)
	hyponyms = defaultdict(list)
	for i,hypo in all_idx.items():
		query_emb = np.expand_dims(all_embedded[i], axis=0)
		search_k = k*4
		hypers = _find_NNs(hypo, query_emb, all_tree, all_idx, k=search_k)

		ct_hyper, ct_hypo = 0, 0
		for hyper, score in hypers:
			if ct_hyper < k and hyper in graph_preds:
				hypernyms[hypo].append((hyper, score))
				ct_hyper += 1

			if ct_hypo < k and hypo in graph_preds:
				hyponyms[hyper].append((hypo, score))
				ct_hypo += 1

	hyponyms = {hyper:sorted(hypos, reverse=True, key=lambda x: x[1]) for hyper, hypos in hyponyms.items()}

	# assert all(p in hypernyms for p in target_preds)

	return hypernyms, hyponyms

def main():
	global ARGS
	ARGS = parser.parse_args()

	utils.print_BAR()
	print(f'Building psuedo-hypernymy tree for EG using KNN, K={ARGS.search_K}')
	utils.checkpoint()
	utils.print_bar()

	print('Reading in dataset...')
	Ent_list, answers, _ = read_dataset('sl_ant', ARGS.data_folder, test=True, directional=True)

	typings = {'#'.join(sorted(e.basic_types)) for pair in Ent_list for e in pair or []}
	print(f'Building trees for {len(typings)} typings')
	print(typings)

	deducer = GraphDeducer('roberta', None)

	graph_trees = {}
	for i,typing in enumerate(typings):
		utils.print_progress(i/len(typings), f'computing {typing}')
		res = load_graph_type(typing, ARGS.graph_embs)
		if not res:
			continue
		inv_idx, embs, searchtree = res
		query_preds = list(set([e.pred_desc() for pair in Ent_list for e in (pair or []) if '#'.join(sorted(e.basic_types)) == typing]))
		hypernyms, hyponyms = compute_hierarchy(typing, query_preds, deducer, inv_idx, embs, searchtree)
		graph_trees[typing] = (hypernyms, hyponyms)
	utils.print_progress(1, 'done')

	utils.print_bar()
	fname = ARGS.dest_fname + '.pkl'
	with open(fname, 'wb+') as f:
		pickle.dump(graph_trees, f, pickle.HIGHEST_PROTOCOL)
		print('Hypernymy trees saved to', fname)

	utils.checkpoint()
	print('done')

parser = argparse.ArgumentParser(description='Generate cache of embeddings for entailment graph nodes')
parser.add_argument('data_folder', help='Path to data folder including freebase entity types and predicate substitution pairs')
parser.add_argument('graph_embs', help='Path to embedded entailment graphs to assist question answering')
parser.add_argument('dest_fname', help='Path to destination file of the embeddings')
parser.add_argument('--search_K', default=10, help='Number of neighbors in graph to identify')
# parser.add_argument('target_dataset', required=True, default='levy_holt', choices=['levy_holt', 'sl_ant'], help='Dataset name to evaluate on')


if __name__ == '__main__':
	main()