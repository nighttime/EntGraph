import argparse
import csv
import sys
import os
import re
from collections import defaultdict, Counter, namedtuple
from operator import itemgetter, attrgetter
import itertools
import random
import datetime
import numpy as np
from sklearn import metrics

import utils
from proposition import *
from entailment import *
from article import *
from reference import bar, BAR

from typing import *

MIN_OCCURRENCE_NEG = 3


def run_test(test_set: List[Tuple[str, str]], graphs: EGraphCache):
	cum_score_neg = 0
	total_tests_neg = 0
	found_tests_neg = 0

	cum_score_pos = 0
	total_tests_pos = 0
	found_tests_pos = 0

	for antecedent, entailment in test_set:
		typing = '#'.join([t.replace('_1', '').replace('_2', '') for t in antecedent.split('#')[1:]])
		neg_ant = antecedent.startswith('NEG')
		if neg_ant:
			total_tests_neg += 1
		else:
			total_tests_pos += 1

		if typing not in graphs:
			print('typing {} not found in graphs'.format(typing))
			continue

		ants = graphs[typing].get_antecedents(entailment)
		for ant in ants:
			if ant.pred == antecedent:
				if neg_ant:
					cum_score_neg += ant.score
					found_tests_neg += 1
				else:
					cum_score_pos += ant.score
					found_tests_pos += 1

	total_tests = total_tests_pos + total_tests_neg
	found_tests = found_tests_pos + found_tests_neg

	print('Results: {} tests, {:.1f}% found edges'.format(total_tests, found_tests/total_tests*100))
	print()
	print('NEG_X => X : {:.1f}% found edges'.format(found_tests_neg/total_tests_neg*100))
	print('{:.4f} avg score / total'.format(cum_score_neg / total_tests_neg))
	print('{:.4f} avg score / found'.format(cum_score_neg / found_tests_neg))
	print()
	print('X => NEG_X : {:.1f}% found edges'.format(found_tests_pos/total_tests_pos*100))
	print('{:.4f} avg score / total'.format(cum_score_pos / total_tests_pos))
	print('{:.4f} avg score / found'.format(cum_score_pos / found_tests_pos))

def generate_test_set(props: List[Prop]) -> List[Tuple[str, str]]:
	negs = Counter()
	for p in props:
		if p.pred_desc().startswith('NEG') and '_2' not in p.types[0]:
			negs[p.pred_desc()] += 1

	filtered_negs = [p for p,ct in negs.items() if ct >= MIN_OCCURRENCE_NEG]

	test_set = []
	for neg_pred in filtered_negs:
		pos_pred = neg_pred[len('NEG__'):]
		test_set.append((neg_pred, pos_pred))
		test_set.append((pos_pred, neg_pred))

	return test_set

def read_test_set(test_location: str) -> List[Tuple[str, str]]:
	tests = []
	with open(test_location) as f:
		reader = csv.reader(f, delimiter='\t')
		for row in reader:
			tests.append(tuple(row))

	return tests

def write_test_set(test_set: List[Tuple[str, str]], filepath):
	with open(filepath, 'w+') as f:
		for ant, ent in test_set:
			f.write('{}\t{}\n'.format(ant, ent))

def main():
	global ARGS
	ARGS = parser.parse_args()

	if bool(ARGS.use_tests) == (bool(ARGS.news_gen_file) and bool(ARGS.data_folder)):
		print('Need the right args!')
		parser.print_help()
		exit(1)

	if ARGS.use_tests:
		test_set = read_test_set(ARGS.use_tests)
	else:
		print('Loading entity type cache...')
		load_precomputed_entity_types(ARGS.data_folder)

		# Read in news articles
		print('Reading source articles & auxiliary data...')
		articles, unary_props, binary_props = read_source_data(ARGS.news_gen_file)

		# Generate test set
		print('Generating test set...')
		test_set = generate_test_set(binary_props)

	if ARGS.write_tests:
		print('Writing test set to file...')
		write_test_set(test_set, ARGS.write_tests)

	if ARGS.graphs:
		# Load entailment graphs
		print('Reading EGs from cache...')
		graphs = read_precomputed_EGs(ARGS.graphs)
		print('Running test...')
		run_test(test_set, graphs)


parser = argparse.ArgumentParser(description='Evaluate the overlap of X and NEG_X predicates')
parser.add_argument('--news-gen-file', help='Path to file used for generating the test set')
parser.add_argument('--data-folder', help='Path to data folder including freebase entity types and predicate substitution pairs')
parser.add_argument('--graphs', help='Path to pre-cached entailment graphs to test')
parser.add_argument('--write-tests', help='output the test set as a .tsv file')
parser.add_argument('--use-tests', help='Use the test file in the given path')

if __name__ == '__main__':
	main()