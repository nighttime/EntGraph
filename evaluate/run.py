import argparse
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
from analyze import *
from questions import *
from answer_wh import *
from answer_tf import *
import ppdb_reader

from typing import *


# For printing
BAR_LEN = 50
BAR = '=' * BAR_LEN
bar = '-' * BAR_LEN

# Program Hyperparameters
# Fraction of props partitioned into Q
QA_SPLIT = 0.5
# TOP_K_ENTS = 100

ARGS = None


def eval_wh_sets(A_list: List[List[Any]], predictions_list: List[List[Any]],
			  eval_fun: Callable[[List[Set[str]], List[List[str]]], float]) -> float:
	A_list_flat, Predictions_list_flat = flatten_answers(A_list, predictions_list)
	return eval_fun(A_list_flat, Predictions_list_flat)

def eval_tf_sets(A_list: List[List[Any]], predictions_list: List[List[Any]],
			  eval_fun: Callable[[List[int], List[int]], float]) -> float:
	A_list_flat, Predictions_list_flat = flatten_answers(A_list, predictions_list)
	return eval_fun(A_list_flat, Predictions_list_flat)


def run(questions, answers, evidence, score_text, uu_graphs=None, bu_graphs=None, eval_fun=ans):
	print('NOT SAFE')
	exit(1)
	print(bar)
	predictions = answer_questions(questions, evidence, uu_graphs=uu_graphs, bu_graphs=bu_graphs)
	score = eval_fun(answers, predictions)
	print(score_text.format(score))
	utils.checkpoint()


def run_wh_sets(Q_list: List[List[str]], A_list: List[List[Set[str]]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
				score_text: str, eval_fun: Callable[[List[Set[str]], List[List[str]]], float],
				uu_graphs: EGraphCache=None, bu_graphs: EGraphCache=None):
	print(bar)
	predictions_list, _ = answer_question_sets(Q_list, evidence_list, uu_graphs=uu_graphs, bu_graphs=bu_graphs, A_list=A_list)
	score = eval_wh_sets(A_list, predictions_list, eval_fun=eval_fun)
	print(score_text)
	print('{:.4f}'.format(score))
	utils.checkpoint()

def run_tf_sets(Q_list: List[List[Prop]], A_list: List[List[int]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
				score_text: str, eval_fun: Callable[[List[Any], List[Any]], float],
				uu_graphs: Optional[EGraphCache]=None,
				bu_graphs: Optional[EGraphCache]=None,
				sim_cache: Optional[EmbeddingCache]=None) -> Tuple[List[List[float]], List[List[Dict[str, Prop]]]]:
	print(bar)
	print(score_text)
	scores_list, supports_list = answer_tf_sets(Q_list, evidence_list, uu_graphs=uu_graphs, bu_graphs=bu_graphs, sim_cache=sim_cache)
	scores_flat = [s for ss in scores_list for s in ss]

	##################################################################
	cutoff = 0
	predictions_list = threshold_sets(scores_list, cutoff, strict=True)
	p = eval_tf_sets(A_list, predictions_list, eval_fun=calc_precision)
	r = eval_tf_sets(A_list, predictions_list, eval_fun=calc_recall)
	a = eval_tf_sets(A_list, predictions_list, eval_fun=acc)
	print('Measurements @{:.2f} score cutoff'.format(cutoff))
	print('precision: {:.3f}'.format(p))
	print('recall:    {:.3f}'.format(r))
	# bins = np.arange(0, 1.01, 0.05)
	# hist = np.histogram(scores_flat, bins=bins)
	# print(hist)
	##################################################################

	# num_answered = len(list(filter(None, scores_flat)))
	# print('Answered {}/{} questions ({:.1f}%)'.format(num_answered, len(scores_flat), num_answered/len(scores_flat)*100))

	utils.checkpoint()
	return scores_list, supports_list



def composite_results(results_UU: Tuple[List[List[float]], List[List[Dict[str, Prop]]]],
					  results_BU: Tuple[List[List[float]], List[List[Dict[str, Prop]]]],
					  adjust_UU: Optional[str],
					  A_list: List[List[int]]) -> Tuple[List[List[float]], List[List[Dict[str, Prop]]]]:
	scores = []
	supports = []

	UU_only = 0
	BU_only = 0
	both = 0
	total = 0

	for i,q_set in enumerate(results_UU[0]):
		scores.append([])
		supports.append([])
		for j,q in enumerate(q_set):
			score_UU = results_UU[0][i][j]
			if not reference.RUNNING_LOCAL and adjust_UU is not None:
				# Remap score based on linear regression to BU model
				if adjust_UU =='linear':
					score_UU = 2.880 * score_UU - 0.056
				if adjust_UU == '3rd':
					# score_UU = max(score_UU, 2.880 * score_UU - 0.056)
					score_UU = -1.523 * (score_UU) + 42.139 * (score_UU * score_UU) - 91.058 * (score_UU * score_UU * score_UU) - 0.017
			score_BU = results_BU[0][i][j]
			score = max(score_UU, score_BU)
			scores[-1].append(score)

			support_UU = results_UU[1][i][j]
			support_BU = results_BU[1][i][j]
			support = {**support_UU, **support_BU}
			supports[-1].append(support)

			if A_list[i][j] == 1:
				total += 1
				if score_UU > 0 and score_BU > 0:
					both += 1
				elif score_UU > 0:
					UU_only += 1
				elif score_BU > 0:
					BU_only += 1

	answered = UU_only + BU_only + both
	print('* Score compositing: {} UU-only, {} BU-only, {} overlap : Answered {}/{} questions ({:.1f}%)'.format(UU_only, BU_only, both, answered, total, answered/total*100))
	return scores, supports


def main():
	global ARGS
	ARGS = parser.parse_args()

	eval_fun = globals()[ARGS.eval_fun]

	print()
	print(BAR)
	mode = 'local' if reference.RUNNING_LOCAL else 'server'
	print('Running Eval ({})'.format(mode))

	original_types_file = 'data/freebase_types/entity2Types.txt'
	print('* Using entity types and substitution pairs from: ' + ARGS.data_folder)

	resources = ['literal answers']
	if ARGS.uu_graphs is not None:
		resources.append('U->U graphs')
	if ARGS.bu_graphs is not None:
		resources.append('B->U graphs')
	if ARGS.sim_cache:
		resources.append('Similarity cache')
	print('* Using evidence from:', str(resources))

	utils.checkpoint()
	print(BAR)

	#######################################

	# Read in U->U and/or B->U entailment graphs
	def load_graphs(graph_dir, message) -> Optional[EGraphCache]:
		if graph_dir:
			print(message, end=' ', flush=True)
			if ARGS.text_EGs:
				graph_ext = 'sim' if ARGS.local else 'binc'
				graphs = read_graphs(graph_dir, ext=graph_ext)
			else:
				graphs = read_precomputed_EGs(graph_dir)
			print('Graphs read: {}'.format(len(graphs)))
			return graphs

	uu_graphs = load_graphs(ARGS.uu_graphs, 'Reading U->U graphs...')
	if uu_graphs and len(uu_graphs) > 3 and mode != 'server':
		print('! Mode is not set properly. Remember to turn off local mode!')
		exit(1)
	bu_graphs = load_graphs(ARGS.bu_graphs, 'Reading B->U graphs...')

	utils.checkpoint()
	print(BAR)

	# Read in entity type cache
	print('Loading entity type cache...')
	# if not os.path.exists(ARGS.data_folder):
	# 	print('  !generating new cache...')
	# 	load_entity_types(original_types_file)
	# 	save_precomputed_entity_types(ARGS.data_folder)
	load_precomputed_entity_types(ARGS.data_folder)

	# Read in similarity cache
	sim_cache = None
	if ARGS.sim_cache:
		print('Loading similarity cache...')
		sim_cache = load_similarity_cache(ARGS.data_folder)

	# Read in news articles
	print('Reading source articles & auxiliary data...')
	articles, unary_props, binary_props = read_source_data(ARGS.news_gen_file)
	negative_swaps = read_substitution_pairs(os.path.join(ARGS.data_folder, 'substitution_pairs.json'))
	ppdb = None
	if ARGS.filter_qs:
		print('Reading PPDB data...')
		ppdb = ppdb_reader.load_ppdb(ARGS.data_folder)

	# Generate questions from Q
	print('Generating questions...', end=' ', flush=True)
	P_list, N_list, evidence_list, top_pred_cache = generate_tf_question_sets(articles, negative_swaps=negative_swaps, uu_graphs=uu_graphs, filter_dict=ppdb)
	num_Qs = sum(len(qs) for qs in P_list + N_list)
	num_P_Qs = sum(len(qs) for qs in P_list)
	num_sets = len(P_list)
	num_articles = len(articles)
	pct_positive = num_P_Qs/num_Qs
	print('Generated {} questions ({:.1f}% +) from {} sets ({} articles)'.format(num_Qs, pct_positive*100, num_sets, num_articles))

	if ARGS.save_qs:
		utils.write_questions_to_file(P_list, N_list)

	if ARGS.save_preds:
		utils.write_pred_cache_to_file(top_pred_cache)

	Q_list, A_list = format_tf_QA_sets(P_list, N_list)

	if ARGS.no_answer:
		print('Quitting before answering phase.')
		utils.checkpoint()
		print(BAR)
		exit(0)

	# Answer the questions using available resources: A set, U->U Graph, B->U graph
	print('Predicting answers...')

	results = {'*always-true':(pct_positive, None)}

	if ARGS.test_all:
		results['*exact-match'] = run_tf_sets(Q_list, A_list, evidence_list, 'form-dependent baseline', eval_fun=eval_fun)

		if uu_graphs:
			results['U->U'] = run_tf_sets(Q_list, A_list, evidence_list, 'U->U', eval_fun=eval_fun, uu_graphs=uu_graphs)

		if bu_graphs:
			results['B->U'] = run_tf_sets(Q_list, A_list, evidence_list, 'B->U', eval_fun=eval_fun, bu_graphs=bu_graphs)

		if uu_graphs and bu_graphs:
			# results['U->U and B->U'] = run_tf_sets(Q_list, A_list, evidence_list, 'U->U and B->U', eval_fun=eval_fun,
			# 			uu_graphs=uu_graphs, bu_graphs=bu_graphs)
			print(bar)
			print('U->U and B->U')
			# results['U->U and B->U'] = composite_results(results['U->U'], results['B->U'], None, A_list)
			# results['U->U and B->U (Adj)'] = composite_results(results['U->U'], results['B->U'], 'linear', A_list)
			# results['U->U and B->U (Adj-3rd)'] = composite_results(results['U->U'], results['B->U'], '3rd', A_list)
			utils.checkpoint()

		if sim_cache:
			results['Similarity'] = run_tf_sets(Q_list, A_list, evidence_list, 'Similarity', eval_fun=eval_fun, sim_cache=sim_cache)

	else:
		results['*exact-match'] = run_tf_sets(Q_list, A_list, evidence_list, 'form-dependent baseline', eval_fun=eval_fun)
		if sim_cache:
			label = 'Similarity'
			results[label] = run_tf_sets(Q_list, A_list, evidence_list, label, eval_fun=eval_fun, sim_cache=sim_cache)
		elif uu_graphs or bu_graphs:
			label = ('U->U' if uu_graphs else '') + (' and ' if uu_graphs and bu_graphs else '') + ('B->U' if bu_graphs else '')
			results[label] = run_tf_sets(Q_list, A_list, evidence_list, label, eval_fun=eval_fun, uu_graphs=uu_graphs, bu_graphs=bu_graphs)

	if ARGS.plot:
		print(bar)
		plot_results(ARGS.data_folder, A_list, results, sample=ARGS.sample, Q_list=Q_list, save_thresh=ARGS.save_thresh)

	utils.checkpoint()
	print(BAR)
	print()



parser = argparse.ArgumentParser(description='Evaluate using P&D style question-generation and -answering')
parser.add_argument('news_gen_file', help='Path to file used for partition into Question set and Answer set')
parser.add_argument('data_folder', help='Path to data folder including freebase entity types and predicate substitution pairs')
parser.add_argument('--uu-graphs', help='Path to Unary->Unary entailment graphs to assist question answering')
parser.add_argument('--bu-graphs', help='Path to Binary->Unary entailment graphs to assist question answering')
parser.add_argument('--sim-cache', action='store_true', help='Use a similarity cache to assist question answering (file must be located in data folder)')
parser.add_argument('--filter-qs', action='store_true', help='Use PPDB to filter questions during generation (file must be located in data folder)')
parser.add_argument('--test-all', action='store_true', help='Test all variations of the given configuration')
parser.add_argument('--text-EGs', action='store_true', help='Read in plain-text entailment graphs from a folder')
parser.add_argument('--local', action='store_true', help='Read in local entailment graphs (default is global)')
parser.add_argument('--eval-fun', default='acc', help='Evaluate results using the specified test function')
parser.add_argument('--day-range', type=int, default=0, help='Evidence window around the day questions are drawn from (extends # days in both directions)')
parser.add_argument('--plot', action='store_true', help='Plot results visually')
parser.add_argument('--sample', action='store_true', help='Sample results at specified precision cutoff')

parser.add_argument('--save-thresh', action='store_true', help='Save precision-level cutoffs for entailment graphs')
parser.add_argument('--save-qs', action='store_true', help='Save generated questions to file')
parser.add_argument('--save-preds', action='store_true', help='Save top predicates to file')

parser.add_argument('--no-answer', action='store_true', help='Stop execution before answering questions')

if __name__ == '__main__':
	main()
