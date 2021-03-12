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
	cum_score = 0
	total_tests = 0
	found_tests = 0

	print('Testing...')

	for antecedent, entailment in test_set:
		typing = '#'.join(antecedent.split('#')[1:])
		ants = graphs[typing].get_antecedents(entailment)
		for ant in ants:
			if ant.pred == antecedent:
				cum_score += ant.score
				found_tests += 1
		total_tests += 1

	print('Results: {} tests, {:.2f} avg score, {} found ({:.2f}%)'.format(total_tests, cum_score/total_tests, found_tests, found_tests/total_tests))

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

	if ARGS.graph_dir:
		# Load entailment graphs
		print('Reading EGs from cache...')
		graphs = read_precomputed_EGs(ARGS.graph_dir)
		print('Running test...')
		run_test(test_set, graphs)


parser = argparse.ArgumentParser(description='Evaluate the overlap of X and NEG_X predicates')
parser.add_argument('--news-gen-file', help='Path to file used for generating the test set')
parser.add_argument('--data-folder', help='Path to data folder including freebase entity types and predicate substitution pairs')
parser.add_argument('--graph-dir', help='Path to folder containing the entailment graphs to test')
parser.add_argument('--write-tests', help='output the test set as a .tsv file')
parser.add_argument('--use-tests', help='Use the test file in the given path')

if __name__ == '__main__':
	main()