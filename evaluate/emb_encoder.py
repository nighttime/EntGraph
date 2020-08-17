# This file needs to be run in an environment with special packages including pytorch_transformers
import argparse
import sys
import os
from collections import defaultdict, Counter, namedtuple
import pickle
from itertools import chain
import numpy as np
import torch
# from pytorch_transformers import *
from transformers import *

from proposition import *
from article import *
import reference
import utils

from typing import *


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
			pred_idx.append(0)
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

def encode(prop: Prop, arg_idx: int, model: BertModel, tokenizer: BertTokenizer) -> np.ndarray:
	pred, arg, ccg_arg_position = construct_proposition(prop, arg_idx=arg_idx)
	tokens_pred = tokenizer.encode(pred)
	tokens_arg = tokenizer.encode(arg)
	tokens = tokens_arg + tokens_pred if ccg_arg_position == 1 else tokens_pred + tokens_arg

	input_ids = torch.tensor([tokens])
	model_outputs = model(input_ids)
	hidden_states = model_outputs[0].squeeze()  # model outputs
	partial_toks = [tokenizer._convert_id_to_token(t) for t in tokens]
	partial_embs = hidden_states[len(tokens_arg):] if ccg_arg_position == 1 else hidden_states[:len(tokens_pred)]
	emb = np.average(partial_embs.detach().numpy(), axis=0)
	return emb

def find_subsequence(needle: List[Any], haystack: List[Any]) -> Tuple[int, int]:
	for start in range(len(haystack)):
		end = start + len(needle)
		if haystack[start:end] == needle:
			return start, end

def encode_batch(batch: List[Tuple[Any, List[str], List[int]]], model: BertModel, tokenizer: BertTokenizer, cache_map: Dict[Any, int], cache: np.ndarray, map_idx: int):
	_, sent_part_list, pred_idx_list = tuple(zip(*batch))
	batch_size = len(sent_part_list)

	# sent_tokens = tokenizer.batch_encode_plus(sent_part_list, pad_to_max_length=True)['input_ids']
	# pred_tokens = tokenizer.batch_encode_plus(pred_idx_list, add_special_tokens=False)['input_ids']
	# pred_bounds = [find_subsequence(pred_tokens[i], sent_tokens[i]) for i in range(batch_size)]

	# sent_tokens = tokenizer.batch_encode_plus(sent_part_list, pad_to_max_length=True)['input_ids']
	sent_tokens = tokenizer.batch_encode_plus([' '.join(ps) for ps in sent_part_list], pad_to_max_length=True)['input_ids']

	input_ids = torch.tensor(sent_tokens)
	model_outputs = model(input_ids)
	hidden_states = model_outputs[0].squeeze()  # model outputs

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

		cache_map[key] = map_idx
		cache[map_idx] = emb
		map_idx += 1

def make_emb_cache(unary_props: List[Prop], binary_props: List[Prop], neg_substitutions: Dict[str, Any]) -> Tuple[Dict[Tuple[str, int], int], np.ndarray]:
	# Initialize Bert model
	print('Initializing BERT...', flush=True, end=' ')
	pretrained_weights = 'bert-base-uncased'
	bert_model = BertModel.from_pretrained(pretrained_weights)#, output_hidden_states=True)
	bert_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)#, never_split=v, do_basic_tokenize=False)

	# Build list of negative propositions
	print('Generating negative unaries...', flush=True, end=' ')
	unary_swaps = []
	for p in unary_props:
		if reference.RUNNING_LOCAL and 'person' not in p.types:
			continue
		p_desc = p.pred_desc()
		if p_desc in neg_substitutions:
			neg_preds = neg_substitutions[p_desc]['troponyms']
			swapped_props = [Prop.with_swapped_pred(p, swap) for swap in neg_preds]
			unary_swaps.extend(swapped_props)

	# Initialize function outputs
	print('Initializing cache...', flush=True, end=' ')
	max_props = len(unary_props) + len(unary_swaps) + len(binary_props)
	cache = np.zeros([max_props, 768])
	cache_map = {}

	# Queue up all proposition fragments to be encoded
	all_prop_frags = chain(unary_props,
						   unary_swaps,
						   binary_props)

	# Embed propositions and save them in the cache
	print('Caching proposition encodings...')
	utils.checkpoint()

	completed = 0
	errors = 0
	batch_size = 512
	batch_map = {}

	for i, prop in enumerate(all_prop_frags):
		if 'E' not in prop.entity_types:
			continue

		if reference.RUNNING_LOCAL and 'person' not in prop.basic_types:
			continue

		# if prop.pred.startswith('be.'):
		# 	continue

		# Skip binaries with reverse-typing (this is an artifact due to the graphs)
		if len(prop.types) == 2 and prop.basic_types[0] == prop.basic_types[1]:
			if int(prop.types[0].split('_')[-1]) == 2:
				continue

		key = prop.prop_desc()
		if key in cache_map or key in batch_map:
			continue

		try:
			sent_parts, pred_idx = construct_sentence(prop)
		except:
			errors += 1
			continue

		batch_map[key] = (sent_parts, pred_idx)

		if len(batch_map) == batch_size:
			batch = [(key, s, p) for key, (s,p) in batch_map.items()]
			encode_batch(batch, bert_model, bert_tokenizer, cache_map, cache, map_idx=completed)
			batch_map = {}
			completed += batch_size
			utils.print_progress(i / max_props, '{} / {} ({} errors)'.format(i, max_props, errors))

	cache = cache[:len(cache_map)]
	return cache_map, cache

def write_cache(prop_idx: Dict[Any, int], prop_embeddings: np.ndarray):
	global ARGS
	dest_folder = (lambda x: x if x.endswith('/') else x + '/')(ARGS.data_folder)
	with open(dest_folder + 'test_prop_emb_idx.pkl', 'wb+') as f:
		pickle.dump(prop_idx, f, pickle.HIGHEST_PROTOCOL)
	np.save(dest_folder + 'prop_embs', prop_embeddings)

def main():
	global ARGS
	ARGS = parser.parse_args()

	utils.checkpoint()

	# Read in entity type cache
	print('Loading entity type cache...')
	load_precomputed_entity_types(ARGS.data_folder)
	print('Reading in data...')
	articles, unary_props, binary_props = read_source_data(ARGS.news_gen_file)
	negative_swaps = read_substitution_pairs(os.path.join(ARGS.data_folder, 'substitution_pairs.json'))
	print('Embedding propositions...', flush=True, end=' ')
	prop_idx, prop_embeddings = make_emb_cache(unary_props, binary_props, negative_swaps)
	print('Writing to file...')
	write_cache(prop_idx, prop_embeddings)
	print('Done')


parser = argparse.ArgumentParser(description='Generate cache of embeddings for propositions')
parser.add_argument('news_gen_file', help='Path to file used for partition into Question set and Answer set')
parser.add_argument('data_folder', help='Path to data folder including freebase entity types and predicate substitution pairs')

if __name__ == '__main__':
	main()