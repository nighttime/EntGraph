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
from article import *
import reference
import utils
from entailment import *

from typing import *

class GraphDeducer:
	def __init__(self, model_name: str, graph_embs_dir, valency=2):
		self.graph_embs_dir = graph_embs_dir
		self.graph_embs_cache: Dict[str, Tuple[Dict[int, str], BallTree]] = {}  # np.ndarray]] = {}
		self.nearest_cache = LRUCache(capacity=100000)

		self.valency = valency
		self.model, self.tokenizer = initialize_model(model_name)

		self.log = defaultdict(int)

	def get_nearest_node(self, preds: List[str], k, available_graphs: Set[str], target_typing=None, ablated_preds=[]) -> Optional[Tuple[List[List[str]], List[List[float]]]]:
		# Check basic assumptions first
		if len(preds) == 0:
			return None
		pred_valency = preds[0].count('#')
		assert pred_valency == self.valency

		basic_typings = ['#'.join(sorted(p.split('#')[1:])).replace('_1', '').replace('_2', '') for p in preds]
		basic_typing = basic_typings[0]
		assert all(t == basic_typing for t in basic_typings)

		# Try to find the correctly-typed graph
		correct_typing = target_typing or basic_typing
		if self.valency == 1 and len(correct_typing.split('#')) == 2:
			use_unary_thing = set(correct_typing.split('#')) == {'thing'}
			emb_result = self.graph_embs_for_basic_typing(correct_typing + '@' + ('thing' if use_unary_thing else basic_typing))
		else:
			emb_result = self.graph_embs_for_basic_typing(correct_typing)

		if not emb_result or correct_typing not in available_graphs:
			# self.log['Graph emb file not found'] += 1
			# return None

			# Back off to the thing#thing graph
			backoff_valency = target_typing.count('#')+1 if target_typing else self.valency
			backoff_typing = '#'.join(['thing']*backoff_valency) #'thing#thing'
			emb_result = self.graph_embs_for_basic_typing(backoff_typing)

			if emb_result:
				self.log['Fallback to {} emb file'.format(backoff_typing)] += 1
				# numbered_typing = 'thing' if self.valency == 1 else '#'.join('thing_' + str(i) for i in range(1,1+self.valency))
				if self.valency == 2:
					numbered_typing = ['thing_1#thing_2']
				elif backoff_valency == 2:
					numbered_typing = ['thing_1', 'thing_2']
				else:
					numbered_typing = ['thing']
				# preds = [p.split('#')[0] + '#' + numbered_typing for p in preds]
				preds = [p.split('#')[0] + '#' + t for p in preds for t in numbered_typing]
			else:
				self.log['Graph emb file not found'] += 1
				return None

		# assert all(p.count('#')==self.valency for p in emb_result[0].values())

		# Check cache for preds to avoid recalculating nearest neighbors
		cached_preds_idx = [i for i, p in enumerate(preds) if (p, basic_typing) in self.nearest_cache]
		cached_results = [self.nearest_cache[(preds[i], basic_typing)] for i in cached_preds_idx]
		preds = [p for i,p in enumerate(preds) if i not in cached_preds_idx]
		if not preds:
			return tuple(zip(*cached_results))

		inv_idx, graph_embs = emb_result

		# Use mask to keep only same-valency candidate replacements
		# keep_inv_idx = {i:p for i,p in inv_idx.items() if p.count('#') == valency}
		# keep_idx = sorted(list(keep_inv_idx.keys()))
		# graph_embs = graph_embs[keep_idx,:]
		# inv_idx = {new_idx:inv_idx[old_idx] for new_idx,old_idx in enumerate(keep_idx)}

		# Delete candidates if ablating
		for to_ablate in ablated_preds:
			raise RuntimeError('Ablation disabled: would give inconsistent results due to caching')
		# 	idx = {p:i for i,p in inv_idx.items()}
		# 	if to_ablate in idx:
		# 		# Temporarily delete a predicate node from the graph (embeddings) for this question instance
		# 		pred_i = idx[to_ablate]
		# 		graph_embs = np.delete(graph_embs, pred_i, axis=0) # this does not mutate original graph_embs!
		# 		inv_idx = {(i if i < pred_i else i - 1):p for i,p in inv_idx.items() if i != pred_i}

		graph_sz = len(inv_idx)

		# if graph_sz > 5000:
		# 	self.log['Skipping graph size >5000'] += 1
		# 	return None

		query_emb = self.encode_preds(preds)

		if k == 'logscale':
			k = max(2, round(math.log10(graph_sz)))
		elif k == 'proportional':
			k = max(2, round(graph_sz/100))
		else:
			k = int(k)
		self.log['k={}'.format(k)] += 1

		if not reference.SMOOTHING_SELF_LOOPS:
			k += 1

		# COSINE
		# norm_query_emb = query_emb / np.linalg.norm(query_emb)
		# sims = graph_embs.dot(norm_query_emb.squeeze())
		# nearest_idx = np.argmax(sims)
		# nearest_score = sims[nearest_idx]

		# SQ EUCLIDEAN
		# dists = pairwise_distances(graph_embs, query_emb, metric='sqeuclidean', n_jobs=-1)
		# 1 pred only
		# nearest_idx = np.argmin(dists)
		# sqdist_ = dists[nearest_idx]
		# dist_ = np.sqrt(sqdist_)
		# nearest_score_ = 1 / (1 + dist_)
		# nearest_score = nearest_score_[0]
		# nearest_pred = inv_idx[nearest_idx]

		# K TOP SCORES (Sq Euclidean)
		if False:
			dists = pairwise_distances(graph_embs, query_emb, metric='sqeuclidean', n_jobs=-1).squeeze()
			# scale k if less than graph size
			if dists.shape[0] <= k:
				k = dists.shape[0] - 1
				print('! scaling k to {} for {}'.format(k, basic_typing))

			top_k_inds_unsorted = np.argpartition(dists, k)[:k] # sorted in ascending order (we want the smallest distances)
			top_k_inds = top_k_inds_unsorted[np.argsort(dists[top_k_inds_unsorted])]
			top_k_dists = np.sqrt(dists[top_k_inds])
			# top_k_dists = dists[top_k_inds]
			top_k_scores = (1 / (1 + top_k_dists))
			# top_k_scores = (1 / (1 + top_k_dists)) * (1 / (1 + math.log10(graph_sz)))
			top_k_preds = [inv_idx[i] for i in top_k_inds]

		if graph_embs.data.shape[0] < k:
			k = graph_embs.data.shape[0]
			self.log['! scaling k to {} for {}'.format(k, basic_typing)] += 1

		top_k_dists, top_k_inds = graph_embs.query(query_emb, k=k)
		top_k_scores = (1 / (1 + (top_k_dists ** 2))).tolist()
		top_k_preds = [[inv_idx[i] for i in row] for row in top_k_inds]

		if not reference.SMOOTHING_SELF_LOOPS:
			for i, pred in enumerate(preds):
				# idx = top_k_preds[i].index(pred) if pred in top_k_preds else len(top_k_preds)
				idx = top_k_preds[i].index(pred) if pred in top_k_preds[i] else len(top_k_preds[i])-1 # changed from above 4/28/22 (bug?)
				del top_k_preds[i][idx]
				del top_k_scores[i][idx]

		# Cache calculated results if needed later
		for i,pred in enumerate(preds):
			self.nearest_cache[(pred, basic_typing)] = (top_k_preds[i], top_k_scores[i])

		# Merge in requested cached results
		if cached_results:
			top_k_preds.extend([ps for ps, ss in cached_results])
			top_k_scores.extend([ss for ps, ss in cached_results])

		assert all(p.count('#')==self.valency for ps in top_k_preds for p in ps), 'Predictions not of valency {}: {}'.format(self.valency, top_k_preds)

		return top_k_preds, top_k_scores

	def encode_preds(self, preds) -> np.ndarray:
		batch = [(pred, *construct_sentence(Prop.from_descriptions(pred, pred.split('#')[1:]))) for pred in preds]
		cache_map = {}
		if len(preds) > 200:
			print('! encoding {} preds'.format(len(preds)))
		cache = np.ndarray([len(preds), 768])
		encode_batch(batch, self.model, self.tokenizer, cache_map, cache, 0)
		return cache

	def graph_embs_for_basic_typing(self, query_typing) -> Optional[Tuple[Dict[int, str], BallTree]]:
		target_typing = ''
		basic_typing = query_typing
		if '@' in query_typing:
			basic_typing, target_typing = query_typing.split('@')

		reverse_typing = '#'.join(reversed(basic_typing.split('#')))

		suffix = '@' + target_typing if target_typing else ''
		query = basic_typing + suffix
		rev_query = reverse_typing + suffix

		if query in self.graph_embs_cache:
			return self.graph_embs_cache[query]
		elif rev_query in self.graph_embs_cache:
			return self.graph_embs_cache[rev_query]
		else:
			embs_path = os.path.join(self.graph_embs_dir, basic_typing + '.pkl')
			if not os.path.exists(embs_path):
				embs_path = os.path.join(self.graph_embs_dir, reverse_typing + '.pkl')
			if not os.path.exists(embs_path):
				return None
			with open(embs_path, 'rb') as f:
				idx, embs = pickle.load(f)

				nonzero_rows = ~(embs==0).all(1)
				embs = embs[nonzero_rows]
				inv_idx = {v:k for k,v in idx.items() if nonzero_rows[v]}

				# Use mask to keep only correct-valency candidate replacements
				keep_inv_idx = {i: p for i, p in inv_idx.items() if p.count('#') == self.valency}
				if target_typing:
					keep_inv_idx = {i: p for i, p in keep_inv_idx.items() if '#'.join(p.split('#')[1:]).replace('_1', '').replace('_2', '') == target_typing}
				keep_idx = sorted(list(keep_inv_idx.keys()))
				embs = embs[keep_idx, :]
				inv_idx = {new_idx: inv_idx[old_idx] for new_idx, old_idx in enumerate(keep_idx)}

				# preprocessing for cosine similarity (norming)
				# norms = np.expand_dims(np.linalg.norm(embs, axis=1), axis=1)
				# embs = embs / norms

				if embs.shape[0] == 0:
					return None

				assert not np.isnan(embs).any()

				embs = BallTree(embs)

				self.graph_embs_cache[query] = (inv_idx, embs)
				self.graph_embs_cache[rev_query] = (inv_idx, embs)
				assert all(p.count('#') == self.valency for p in inv_idx.values())
				return inv_idx, embs

	# def graph_embs_for_basic_typing(self, query_typing) -> Optional[Tuple[Dict[int, str], BallTree]]:
	# 	target_typing = ''
	# 	if '@' in query_typing:
	# 		basic_typing, target_typing = query_typing.split('@')
	# 		target_typing = '@' + target_typing
	#
	# 	reverse_typing = '#'.join(reversed(basic_typing.split('#')))
	# 	if basic_typing in self.graph_embs_cache:
	# 		return self.graph_embs_cache[basic_typing]
	# 	elif reverse_typing in self.graph_embs_cache:
	# 		return self.graph_embs_cache[reverse_typing]
	# 	else:
	# 		embs_path = os.path.join(self.graph_embs_dir, basic_typing + '.pkl')
	# 		if not os.path.exists(embs_path):
	# 			embs_path = os.path.join(self.graph_embs_dir, reverse_typing + '.pkl')
	# 		if not os.path.exists(embs_path):
	# 			return None
	# 		with open(embs_path, 'rb') as f:
	# 			idx, embs = pickle.load(f)
	#
	# 			nonzero_rows = ~(embs == 0).all(1)
	# 			embs = embs[nonzero_rows]
	# 			inv_idx = {v: k for k, v in idx.items() if nonzero_rows[v]}
	#
	# 			# Use mask to keep only correct-valency candidate replacements
	# 			keep_inv_idx = {i: p for i, p in inv_idx.items() if p.count('#') == self.valency}
	# 			keep_idx = sorted(list(keep_inv_idx.keys()))
	# 			embs = embs[keep_idx, :]
	# 			inv_idx = {new_idx: inv_idx[old_idx] for new_idx, old_idx in enumerate(keep_idx)}
	#
	# 			# preprocessing for cosine similarity (norming)
	# 			# norms = np.expand_dims(np.linalg.norm(embs, axis=1), axis=1)
	# 			# embs = embs / norms
	#
	# 			if embs.shape[0] == 0:
	# 				return None
	#
	# 			assert not np.isnan(embs).any()
	#
	# 			embs = BallTree(embs)
	#
	# 			self.graph_embs_cache[basic_typing] = (inv_idx, embs)
	# 			self.graph_embs_cache[reverse_typing] = (inv_idx, embs)
	# 			assert all(p.count('#') == self.valency for p in inv_idx.values())
	# 			return inv_idx, embs

# Based on: https://www.geeksforgeeks.org/lru-cache-in-python-using-ordereddict/
class LRUCache:
	def __init__(self, capacity: int):
		self.cache = OrderedDict()
		self.capacity = capacity

	def get(self, key):
		# if key not in self.cache:
		# 	return None
		# else:
		# 	self.cache.move_to_end(key)
		# 	return self.cache[key]
		value = self.cache[key]
		self.cache.move_to_end(key)
		return value

	def put(self, key: int, value: int) -> None:
		self.cache[key] = value
		self.cache.move_to_end(key)
		if len(self.cache) > self.capacity:
			self.cache.popitem(last=False)

	def __contains__(self, item):
		return item in self.cache

	def __getitem__(self, item):
		return self.get(item)

	def __setitem__(self, key, value):
		self.put(key, value)

def construct_proposition(prop: Prop, arg_idx=0) -> Tuple[str, str, int]:
	is_unary = (len(prop.args) == 1)
	arg = prop.args[arg_idx]

	# Break down predicate into constituent parts and extract CCG arg position
	if is_unary:
		pred_parts = prop.pred.split('.')
	else:
		binary_pred_halves = prop.pred[prop.pred.find('(')+1:prop.pred.find(')')].split(',')
		pred_parts = binary_pred_halves[arg_idx].split('.')

	ccg_arg_position = int(pred_parts[-1])
	pred_parts = pred_parts[:-1]

	# Rephrase certain structures
	if is_unary and len(pred_parts) > 1 and pred_parts[0] == 'be' and ccg_arg_position == 2:
		pred_parts = pred_parts[1:-1] + [pred_parts[0], pred_parts[-1]]

	if 'be' in pred_parts:
		pred_parts = ['is' if p == 'be' else p for p in pred_parts]

	# Assemble predicate
	predicate = ' '.join(pred_parts)

	# Assemble sentence
	parts = [arg, predicate] if ccg_arg_position == 1 else [predicate, arg]
	sent = ' '.join(parts)

	return sent, predicate, ccg_arg_position

def pred_deoverlap(pred_parts_l: List[str], pred_parts_r: List[str]) -> Tuple[List[str], List[str]]:
	left, right = [], []

	i = 0
	while i < len(pred_parts_l) and i < len(pred_parts_r) and pred_parts_l[i] == pred_parts_r[i]:
		i += 1

	left = pred_parts_l
	right = pred_parts_r[i:]

	return left, right

# Returns the constructed sentence chunked into predicate and arguments
def construct_sentence(prop: Prop) -> Tuple[List[str], List[int]]:
	is_unary = (len(prop.args) == 1)

	if is_unary:
		pred_parts = prop.pred.split('.')
		ccg_arg_position = int(pred_parts[-1])

		# Special case: be.president.2
		if len(pred_parts) > 1 and pred_parts[0] == 'be' and ccg_arg_position == 2:
			pred_parts = pred_parts[1:-1] + [pred_parts[0], pred_parts[-1]]

		if 'be' in pred_parts:
			pred_parts = ['is' if p == 'be' else p for p in pred_parts]

		predicate = ' '.join(pred_parts[:-1])
		arg = prop.args[0]
		sent_parts = [arg, predicate] if ccg_arg_position == 1 else [predicate, arg]
		pred_idx = [0, 0]
		pred_idx[sent_parts.index(predicate)] = 1
	else:
		binary_pred_halves = prop.pred[prop.pred.find('(')+1:prop.pred.find(')')].split(',')
		pred_parts_l, pred_parts_r = binary_pred_halves[0].split('.'), binary_pred_halves[1].split('.')

		if 'be' in pred_parts_l:
			pred_parts_l = ['is' if p == 'be' else p for p in pred_parts_l]

		if 'be' in pred_parts_r:
			pred_parts_r = ['is' if p == 'be' else p for p in pred_parts_r]

		ccg_arg_position_l = int(pred_parts_l[-1])

		pred_l, pred_r = pred_deoverlap(pred_parts_l[:-1], pred_parts_r[:-1])
		assert len(pred_l) > 0

		sent_parts = []
		pred_idx = []
		if ccg_arg_position_l == 1:
			sent_parts.append(prop.args[0])
			sent_parts.append(' '.join(pred_l))
			pred_idx.append(0)
			pred_idx.append(1)
		else:
			sent_parts.append(' '.join(pred_l))
			sent_parts.append(prop.args[0])
			pred_idx.append(1)
			pred_idx.append(0)

		if pred_r:
			sent_parts.append(' '.join(pred_r))
			pred_idx.append(1)

		sent_parts.append(prop.args[1])
		pred_idx.append(0)

	assert len(sent_parts) == len(pred_idx)

	return sent_parts, pred_idx

def find_subsequence(needle: List[Any], haystack: List[Any]) -> Tuple[int, int]:
	for start in range(len(haystack)):
		end = start + len(needle)
		if haystack[start:end] == needle:
			return start, end

def encode_batch(batch: List[Tuple[Any, List[str], List[int]]], model: PreTrainedModel, tokenizer: PreTrainedTokenizer, cache_map: Dict[Any, int], cache: np.ndarray, map_idx: int) -> int:
	_, sent_part_list, pred_idx_list = tuple(zip(*batch))
	batch_size = len(sent_part_list)

	# sent_tokens = tokenizer.batch_encode_plus(sent_part_list, pad_to_max_length=True)['input_ids']
	# pred_tokens = tokenizer.batch_encode_plus(pred_idx_list, add_special_tokens=False)['input_ids']
	# pred_bounds = [find_subsequence(pred_tokens[i], sent_tokens[i]) for i in range(batch_size)]

	# sent_tokens = tokenizer.batch_encode_plus(sent_part_list, pad_to_max_length=True)['input_ids']
	sent_tokens = tokenizer.batch_encode_plus([' '.join(ps) for ps in sent_part_list], pad_to_max_length=True)['input_ids']

	input_ids = torch.tensor(sent_tokens)
	model_outputs = model(input_ids)
	hidden_states = model_outputs[0] #.squeeze()  # model outputs
	errors = 0

	for i, (key, sent_parts, pred_idx) in enumerate(batch):
		start = 1
		ct = 0
		partial_embs = np.zeros(hidden_states[i][0].shape)
		for j, pred_val in enumerate(pred_idx):
			toks = tokenizer.encode(sent_parts[j], add_special_tokens=False)
			num_toks = len(toks)
			if pred_val == 1:
				partial_embs += torch.sum(hidden_states[i][start:start+num_toks], dim=0).detach().numpy()
				ct += num_toks
			start += num_toks

		# emb = np.average(partial_embs.detach().numpy(), axis=0)
		emb = partial_embs / ct

		if any(np.isnan(emb)):
			print('\nError: NaN occurred while encoding:', key, sent_parts, str(pred_idx))
			errors += 1
			continue

		cache_map[key] = map_idx
		cache[map_idx] = emb
		map_idx += 1

	return errors

def initialize_model(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
	model = None
	tokenizer = None
	if model_name == 'bert':
		print('Initializing BERT...')
		pretrained_weights = 'bert-base-uncased'
		model = BertModel.from_pretrained(pretrained_weights)  # , output_hidden_states=True)
		tokenizer = BertTokenizer.from_pretrained(pretrained_weights)  # , never_split=v, do_basic_tokenize=False)
	elif model_name == 'roberta':
		print('Initializing ROBERTA...')
		pretrained_weights = 'roberta-base'
		model = RobertaModel.from_pretrained(pretrained_weights)
		tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
	return model, tokenizer


def embed_graph(graph: EntailmentGraph, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[Dict[Tuple[str, int], int], np.ndarray]:
	assert graph.edges

	# graph_valency = len(graph.typing.split('#'))
	# preds = list(graph.edges.keys())
	preds = list(graph.nodes)
	preds = [p for p in preds if (not article.reject_unary(p) if p.count('#') == 1 else not article.reject_binary(p))]
	# if graph_valency == 1:
	# 	preds = [p for p in preds if not article.reject_unary(p)]
	# else:
	# 	preds = [p for p in preds if not article.reject_binary(p)]

	max_props = len(preds)
	cache = np.zeros([max_props, 768])
	cache_map = {}

	batch_size = 512
	completed = 0
	batch = []
	# encode in batches
	for pred in preds:
		typing = pred.split('#')[1:]
		try:
			sent_parts, pred_idx = construct_sentence(Prop.from_descriptions(pred, typing))
			batch.append((pred, sent_parts, pred_idx))
		except:
			continue

		if len(batch) == batch_size:
			encode_batch(batch, model, tokenizer, cache_map, cache, completed)
			batch = []
			completed += batch_size

	# encode remainder ( % batch size)
	if batch:
		encode_batch(batch, model, tokenizer, cache_map, cache, completed)

	return cache_map, cache

def write_cache(dest_folder: str, fname: str, idx: Dict[Any, int], embeddings: np.ndarray):
	with open(os.path.join(dest_folder, fname + '.pkl'), 'wb+') as f:
		pickle.dump([idx, embeddings], f, pickle.HIGHEST_PROTOCOL)


def make_model(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
	model = None
	tokenizer = None
	if model_name == 'bert':
		print('Initializing BERT...')
		pretrained_weights = 'bert-base-uncased'
		model = BertModel.from_pretrained(pretrained_weights)  # , output_hidden_states=True)
		tokenizer = BertTokenizer.from_pretrained(pretrained_weights)  # , never_split=v, do_basic_tokenize=False)
	elif model_name == 'roberta':
		print('Initializing RoBERTa...')
		pretrained_weights = 'roberta-base'
		model = RobertaModel.from_pretrained(pretrained_weights)
		tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)

	return model, tokenizer


def main():
	global ARGS
	ARGS = parser.parse_args()

	utils.checkpoint()

	# print('Loading entailment graphs...')
	graphs = load_graphs(ARGS.graphs, 'Loading graphs...', ARGS)
	print('Read {} BU Graphs'.format(len(graphs)))

	utils.checkpoint()

	dest_folder = ARGS.dest_folder
	if not os.path.exists(dest_folder):
		os.makedirs(dest_folder)

	print('Preparing model')
	model, tokenizer = make_model(ARGS.model)

	print('Embedding graphs in {}'.format(dest_folder))
	seen = set()
	for i,graph in enumerate(graphs.values()):
		if graph.typing in seen:
			continue
		utils.print_progress(i/len(graphs), graph.typing)
		idx, embeddings = embed_graph(graph, model, tokenizer)
		write_cache(dest_folder, graph.typing, idx, embeddings)
		seen.add(graph.typing)
	utils.print_progress(1, 'done')

	utils.checkpoint()

	print('Done')


parser = argparse.ArgumentParser(description='Generate cache of embeddings for entailment graph nodes')
parser.add_argument('data_folder', help='Path to data folder including freebase entity types and predicate substitution pairs')
# parser.add_argument('uu_graphs', help='Path to Unary->Unary entailment graphs to assist question answering')
parser.add_argument('graphs', help='Path to entailment graphs to assist question answering')
parser.add_argument('dest_folder', help='Path to destination folder of the embeddings')
parser.add_argument('--model', required=True, choices=['bert', 'roberta'], default='roberta', help='Choice of model used to encode graph predicates')
parser.add_argument('--text-EGs', action='store_true', help='Read in plain-text entailment graphs from a folder')
parser.add_argument('--local', action='store_true', help='Read in local entailment graphs (default is global)')

if __name__ == '__main__':
	main()