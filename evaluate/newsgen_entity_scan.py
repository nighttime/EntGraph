import argparse
import os
import re
from datetime import datetime
from collections import Counter, defaultdict
import json
from dataclasses import dataclass
import pickle

import utils
from reference import tcolors
from proposition import load_precomputed_entity_types, get_type
from typing import *

def scan_source_data_entities(news_gen_file):
	# reg = re.compile(r'"r":"([^"]*)')
	reg = re.compile(r'"r":"([^}]*)"}')
	entity_idx = {}

	def _get_entity_id(e):
		if e not in entity_idx:
			entity_idx[e] = len(entity_idx)
		return entity_idx[e]

	pred_features: Dict[str, Counter] = defaultdict(Counter)

	# Tally feature counts for each predicate
	print('> Reading news gen file for predicates and entities...')
	with open(news_gen_file) as f:
		for line in f:
			if not line.startswith('{'):
				continue

			# data = json.loads(line)
			# rels_list = data['rels']
			# for r in rels_list:
			# 	rel_str = r['r']
			for match in re.findall(reg, line):
				pred_str = match[1:-1]
				# pred_str = rel_str[1:-1]
				parts = pred_str.split('::')
				pred, arg1, arg2, entity_typing = parts[:4]
				if not any(p in pred for p in ['(buy.1,buy.2)', '(acquire.1,acquire.2)', '(purchase.1,purchase.2)']):
					continue
				if 'E' not in entity_typing:
					continue
				if pred.startswith('LNEG__'):
					pred = pred[1:]
				feature = (_get_entity_id(arg1), _get_entity_id(arg2))
				type1 = get_type(arg1, entity_typing[0] == 'E')
				type2 = get_type(arg2, entity_typing[1] == 'E')
				typed_pred = '{}#{}#{}'.format(pred, type1, type2)
				pred_features[typed_pred][feature] += 1

	# Compare P-N and P-M feature sets
	print('Comparing predicate feature sets...')
	# print('NEG')
	print('MOD')
	for neg_pred, neg_feats in pred_features.items():
		# if not neg_pred.startswith('NEG__'):
		if not neg_pred.startswith('MOD__'):
			continue
		pred = neg_pred[len('NEG__'):]
		if pred not in pred_features.keys():
			continue

		feats = pred_features[pred]

		if not any(t in pred for t in ['#person', '#organization', '#location', '#thing#thing', '#government']):
			continue

		if feats and neg_feats:
			num_common_keys = sum(1 if k in feats else 0 for k in neg_feats.keys())
			print(pred)
			num_instances = sum(v for k,v in feats.items())
			num_neg_instances = sum(v for k,v in neg_feats.items())
			print('P: {} ({})\tN: {} ({})\t'.format(len(feats), num_instances, len(neg_feats), num_neg_instances) +
				  tcolors.BOLD + 'P n N: {:.2f}\t'.format(num_common_keys) + tcolors.ENDC +
				  'n / N: {:.2f}\tn / P: {:.2f}'.format(num_common_keys / len(neg_feats), num_common_keys / len(feats)))
			print()


def main():
	global ARGS, MARGIN
	ARGS = parser.parse_args()
	utils.checkpoint()
	print('Reading entity type data from {} ...'.format(ARGS.data_folder))
	load_precomputed_entity_types(ARGS.data_folder)

	# fpath_news_data = os.path.join(ARGS.data_folder, 'news_data_yago.pkl')
	# if ARGS.read_news:
	# 	print('Reading precomputed source articles from {} ...'.format(fpath_news_data))
	# 	with open(fpath_news_data, 'rb') as f:
	# 		articles = pickle.load(f)
	# 		scan_entities_in_
	# else:
	print('Reading source articles from {} ...'.format(ARGS.news_gen_file))
	scan_source_data_entities(ARGS.news_gen_file)
	utils.checkpoint()

parser = argparse.ArgumentParser(description='Resolving Yago times with News Data')
parser.add_argument('news_gen_file', help='Path to file used for partition into Question set and Answer set')
parser.add_argument('data_folder', help='Path to data folder including freebase entity types and predicate substitution pairs')

if __name__ == '__main__':
	main()