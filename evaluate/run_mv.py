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

import reference
import utils
from graph_encoder import GraphDeducer
from proposition import *
from entailment import *
from article import *
from analyze import *
from questions import *
from answer_wh import *
from answer_tf import *
import ppdb_reader

from typing import *


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
	utils.print_bar()
	predictions = answer_questions(questions, evidence, uu_graphs=uu_graphs, bu_graphs=bu_graphs)
	score = eval_fun(answers, predictions)
	print(score_text.format(score))
	utils.checkpoint()


def run_wh_sets(Q_list: List[List[str]], A_list: List[List[Set[str]]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
				score_text: str, eval_fun: Callable[[List[Set[str]], List[List[str]]], float],
				uu_graphs: EGraphCache=None, bu_graphs: EGraphCache=None):
	utils.print_bar()
	predictions_list, _ = answer_question_sets(Q_list, evidence_list, uu_graphs=uu_graphs, bu_graphs=bu_graphs, A_list=A_list)
	score = eval_wh_sets(A_list, predictions_list, eval_fun=eval_fun)
	print(score_text)
	print('{:.4f}'.format(score))
	utils.checkpoint()

# def run_tf_sets(Q_list: List[List[Prop]], A_list: List[List[int]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
# 				score_text: str, eval_fun: Callable[[List[Any], List[Any]], float],
# 				uu_graphs: Optional[EGraphCache]=None,
# 				bu_graphs: Optional[EGraphCache]=None,
# 				sim_cache: Optional[EmbeddingCache]=None) -> Tuple[List[List[float]], List[List[Dict[str, Prop]]]]:
def run_tf_sets(Q_list: List[List[Prop]], A_list: List[List[int]],
				evidence_list: List[Tuple[List[Prop], List[Prop]]],
				score_text: str, eval_fun: Callable[[List[Any], List[Any]], float],
				models: Dict[str, Any],
				answer_modes: Set[str]) -> Tuple[List[List[float]], List[List[Dict[str, Prop]]]]:
	utils.print_bar()
	print(score_text)
	global ARGS
	scores_list, supports_list, error_cases = answer_tf_sets(Q_list, evidence_list, models, answer_modes)
	# if ARGS.save_errors:
	# 	utils.save_props_on_file(error_cases, ARGS.data_folder, 'sim_hypo_NaN')
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

# def precache_nearest_neighbors_mp(props_list: List[List[Prop]], graph_emb_fname: str) -> Dict[Prop: Optional[Tuple[List[str], List[float]]]]:
# 	def compute_nearest_neighbors_set(props: List[Prop], graph_emb_fname: str):
# 		graph_deducer = GraphDeducer('roberta', graph_emb_fname)
# 		for prop in props:
# 			graph_deducer.get_nearest_node(prop.pred_desc(), k=4, position='left', model='roberta')
# 	cache = {}
# 	graph_fpaths = graph_files_in_dir(graph_dir, None, True)
# 	pool = mp.Pool()
# 	graphs = pool.map(open_graph, props_list)




def run_evaluate() -> Tuple[List[List[Article]],
							List[List[Prop]],
							List[List[int]],
							List[Tuple[List[Prop], List[Prop]]],
							Optional[Dict[str, Tuple[List[List[float]], List[List[Dict[str, Prop]]]]]]]:
	global ARGS
	ARGS = parser.parse_args()

	eval_fun = globals()[ARGS.eval_fun]

	print()
	utils.print_BAR()
	mode = 'local' if reference.RUNNING_LOCAL else 'server'
	title = 'Running Eval ({})'.format(mode)
	if ARGS.quick:
		title += ' (quick mode)'
	print(title)

	if ARGS.test_mode:
		print('* EVAL on TEST SET')

	original_types_file = 'data/freebase_types/entity2Types.txt'
	print('* Using entity types and substitution pairs from: ' + ARGS.data_folder)

	resources = ['literal answers']
	if ARGS.uu_graphs is not None:
		resources.append('U->U graphs')
	if ARGS.bu_graphs is not None:
		resources.append('B->U graphs')
	if ARGS.sim_cache:
		resources.append('Similarity cache')
	if ARGS.ppdb:
		resources.append('PPDB')
	if ARGS.uv_emb is not None:
		resources.append('UV graph embeddings')
	if ARGS.bv_emb is not None:
		resources.append('BV graph embeddings')
	print('* Using evidence from:', str(resources))

	question_modes = set()
	question_modes |= {'unary'}
	question_modes |= {'binary'}

	print('* Question modes:', question_modes)

	answer_modes = set()
	if 'unary' in question_modes and ARGS.uu_graphs:
		answer_modes.add('UU')
	if 'binary' in question_modes and ARGS.bu_graphs:
		answer_modes.add('BB')
		if not ARGS.quick:
			answer_modes.add('BU')

	if ARGS.backoff != 'none':
		reference.GRAPH_BACKOFF = ARGS.backoff
		print('* Answer modes:', answer_modes, '[+Backoff {}]'.format(ARGS.backoff))
	else:
		print('* Answer modes:', answer_modes)

	utils.checkpoint()
	utils.print_BAR()

	#######################################

	models = {}

	# Read in U->U and/or B->U entailment graphs
	def load_graphs(graph_dir, message) -> Optional[EGraphCache]:
		if graph_dir:
			print(message, end=' ', flush=True)
			if ARGS.text_EGs:
				stage = EGStage.LOCAL if ARGS.local else EGStage.GLOBAL
				graphs = read_graphs(graph_dir, stage)
			else:
				graphs = read_precomputed_EGs(graph_dir)

			# Count nodes and edges
			# nodes = 0
			# edges = 0
			# toU_edges = 0
			# for g in set(graphs.values()):
			# 	antecedents = [e.pred for e_set in g.backmap.values() for e in e_set]
			# 	print(g.typing, 'nodes', len(set(antecedents)), 'edges', len(antecedents))
			# 	nodes += len(set(antecedents))
			# 	edges += len(antecedents)
			# 	toU = sum([len(g.backmap[ent]) for ent in g.backmap.keys() if ent.count('#') == 1])
			# 	toU_edges += toU
			#
			# print('Graphs read: {}'.format(len(graphs)))
			# print('antecedents: {} entailments: {}'.format(nodes, edges))
			# print('edges: ? -> U {}'.format(toU_edges))
			return graphs

	uu_graphs = load_graphs(ARGS.uu_graphs, 'Reading U->U graphs...')
	if uu_graphs and len(uu_graphs) > 3 and mode != 'server':
		print('! Mode is not set properly. Remember to turn off local mode!')
		exit(1)
	bu_graphs = load_graphs(ARGS.bu_graphs, 'Reading B->U graphs...')
	print()

	if uu_graphs:
		models['UU'] = uu_graphs
	if bu_graphs:
		models['BU'] = bu_graphs

	utils.checkpoint()

	ppdb = None
	if ARGS.ppdb:
		utils.print_bar()
		print('Reading PPDB...')
		ppdb = ppdb_reader.load_ppdb(ARGS.ppdb)
		models['PPDB'] = ppdb
		utils.checkpoint()

	utils.print_BAR()

	# Read in entity type cache
	print('Loading entity type cache...')
	# if not os.path.exists(ARGS.data_folder):
	# 	print('  !generating new cache...')
	# 	load_entity_types(original_types_file)
	# 	save_precomputed_entity_types(ARGS.data_folder)
	load_precomputed_entity_types(ARGS.data_folder)

	# Read in similarity cache
	if ARGS.sim_cache:
		print('Loading similarity cache...', end=' ', flush=True)
		found = False
		cache_location = ARGS.data_folder
		if ARGS.test_mode:
			cache_location = os.path.join(cache_location, 'test_data')
		try:
			sim_cache_bert = load_similarity_cache(cache_location, 'bert')
			models['BERT'] = sim_cache_bert
			found = True
			print('Loaded BERT...', end=' ', flush=True)
		except:
			print(reference.tcolors.FAIL + 'FAILED to load BERT...' + reference.tcolors.ENDC, end=' ', flush=True)
		try:
			sim_cache_roberta = load_similarity_cache(cache_location, 'roberta')
			models['RoBERTa'] = sim_cache_roberta
			found = True
			print('Loaded RoBERTa...', end=' ', flush=True)
		except:
			print(reference.tcolors.FAIL + 'FAILED to load RoBERTa...' + reference.tcolors.ENDC, end=' ', flush=True)
		print()
		if not found:
			print(reference.tcolors.FAIL + '! No cached model data was found. Quitting.' + reference.tcolors.ENDC)
			exit(1)

	# Read in news articles
	print('Reading source articles & auxiliary data...')
	articles, unary_props, binary_props = read_source_data(ARGS.news_gen_file)
	swap_folder = ARGS.data_folder
	if ARGS.test_mode:
		swap_folder = os.path.join(swap_folder, 'test_data')
	negative_swaps = read_substitution_pairs(os.path.join(swap_folder, 'substitution_pairs.json'))

	# Generate questions from Q
	print('Generating questions...', end=' ', flush=True)
	partitions, P_list, N_list, evidence_list, top_pred_cache = generate_tf_question_sets(articles,
																						  question_modes,
																						  negative_swaps=negative_swaps,
																						  uu_graphs=uu_graphs,
																						  bu_graphs=bu_graphs,
																						  filter_dict=None)
	utils.analyze_questions(P_list, N_list, uu_graphs, bu_graphs, ppdb)
	print()

	if ARGS.graph_qs:
		print('Selecting only questions with hits in the graphs')
		P_list, N_list = select_answerable_qs(P_list, N_list, uu_graphs, bu_graphs)
		utils.analyze_questions(P_list, N_list, uu_graphs, bu_graphs, ppdb)
		print()

	print('Balancing questions...')
	P_list, N_list = rebalance_qs(P_list, N_list, pct_unary=0.5, pct_pos=0.5)
	utils.analyze_questions(P_list, N_list, uu_graphs, bu_graphs, ppdb)
	print()

	# if ARGS.graph_qs:
	# 	print('Selecting only questions with hits in the graphs...')
	# 	P_list, N_list = select_answerable_qs(P_list, N_list, uu_graphs, bu_graphs)
	# 	utils.analyze_questions(P_list, N_list, uu_graphs, bu_graphs)
	# 	print()
	#
	# 	print('Balancing questions...')
	# 	P_list, N_list = rebalance_qs(P_list, N_list, pct_unary=0.5, pct_pos=0.5)
	# 	utils.analyze_questions(P_list, N_list, uu_graphs, bu_graphs)
	# 	print()

	# if reference.RUNNING_LOCAL:
	# 	print('(IGNORE positivity rate for local question generation; filtering is done based on available graphs)')
	if ARGS.save_qs:
		utils.write_questions_to_file(P_list, N_list)
		print('Questions written to file.')

	if ARGS.save_preds:
		utils.write_pred_cache_to_file(top_pred_cache)
		print('Pred cache written to file.')

	Q_list, A_list = format_tf_QA_sets(P_list, N_list)

	if ARGS.no_answer:
		print('Quitting before the answering phase.')
		utils.checkpoint()
		utils.print_BAR()
		return partitions, Q_list, A_list, evidence_list, None

	# Answer the questions using available resources: A set, U->U Graph, B->U graph
	utils.print_BAR()
	print('Predicting answers...')

	# results = {'*always-true':(pct_positive, None)}
	results = {}

	# if ARGS.test_all:
	# results['*exact-match'] = run_tf_sets(Q_list, A_list, evidence_list, 'form-dependent baseline', eval_fun, models, answer_modes)
	if 'unary' in question_modes:
		answer_modes = {'Literal U'}
		results['*exact-match U'] = run_tf_sets(Q_list, A_list, evidence_list, 'Literal U', eval_fun, models, answer_modes)

	if 'binary' in question_modes:
		answer_modes = {'Literal B'}
		results['*exact-match B'] = run_tf_sets(Q_list, A_list, evidence_list, 'Literal B', eval_fun, models, answer_modes)

	if bu_graphs:
		if ARGS.bv_emb:
			print('Using graph embeddings from: {} ([B]->U nodes)'.format(ARGS.bv_emb))
			models['BV-B-Deducer'] = GraphDeducer('roberta', ARGS.bv_emb, valency=2)

		if 'unary' in question_modes and not ARGS.quick:
			if ARGS.bv_emb:
				print('Using graph embeddings from: {} (B->[U] nodes)'.format(ARGS.bv_emb))
				models['BV-U-Deducer'] = GraphDeducer('roberta', ARGS.bv_emb, valency=1)
				answer_modes = {'BU', 'BU-LM', 'Literal U'}
				results['BU-LM'] = run_tf_sets(Q_list, A_list, evidence_list, 'BU-LM', eval_fun, models, answer_modes)
				print('BV-B-Deducer log:', models['BV-B-Deducer'].log)
				print('BV-U-Deducer log:', models['BV-U-Deducer'].log)
			answer_modes = {'BU', 'Literal U'}
			results['BU'] = run_tf_sets(Q_list, A_list, evidence_list, 'BU', eval_fun, models, answer_modes)

		if 'binary' in question_modes:
			if ARGS.bv_emb:
				answer_modes = {'BB', 'BB-LM', 'Literal B'}
				results['BB-LM'] = run_tf_sets(Q_list, A_list, evidence_list, 'BB-LM', eval_fun, models, answer_modes)
				print('BV-B-Deducer log:', models['BV-B-Deducer'].log)
			answer_modes = {'BB', 'Literal B'}
			results['BB'] = run_tf_sets(Q_list, A_list, evidence_list, 'BB', eval_fun, models, answer_modes)

	if uu_graphs and 'unary' in question_modes:
		if ARGS.uv_emb:
			print('Using graph embeddings from: {}'.format(ARGS.uv_emb))
			models['UV-U-Deducer'] = GraphDeducer('roberta', ARGS.uv_emb, valency=1)
			answer_modes = {'UU', 'UU-LM', 'Literal U'}
			results['UU-LM'] = run_tf_sets(Q_list, A_list, evidence_list, 'UU-LM', eval_fun, models, answer_modes)
			print('UV-U-Deducer log:', models['UV-U-Deducer'].log)

		answer_modes = {'UU', 'Literal U'}
		results['UU'] = run_tf_sets(Q_list, A_list, evidence_list, 'UU', eval_fun, models, answer_modes)

	if ARGS.sim_cache:
		if 'BERT' in models:
			answer_modes = {'BERT'}
			results['BERT'] = run_tf_sets(Q_list, A_list, evidence_list, 'BERT', eval_fun, models, answer_modes)
		if 'RoBERTa' in models:
			answer_modes = {'RoBERTa'}
			results['RoBERTa'] = run_tf_sets(Q_list, A_list, evidence_list, 'RoBERTa', eval_fun, models, answer_modes)

	if ARGS.ppdb:
		utils.print_bar()
		answer_modes = {'PPDB'}
		results['PPDB'] = run_tf_sets(Q_list, A_list, evidence_list, 'PPDB', eval_fun, models, answer_modes)

	# else:
	# 	results['*exact-match'] = run_tf_sets(Q_list, A_list, evidence_list, 'form-dependent baseline', eval_fun, models, answer_modes)
	# 	if sim_cache:
	# 		label = 'Similarity'
	# 		results[label] = run_tf_sets(Q_list, A_list, evidence_list, label, eval_fun=eval_fun, sim_cache=sim_cache)
	# 	elif uu_graphs or bu_graphs:
	# 		label = ('U->U' if uu_graphs else '') + (' and ' if uu_graphs and bu_graphs else '') + ('B->U' if bu_graphs else '')
	# 		results[label] = run_tf_sets(Q_list, A_list, evidence_list, label, eval_fun=eval_fun, uu_graphs=uu_graphs, bu_graphs=bu_graphs)

	reference.FINISH_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')

	if ARGS.save_results:
		utils.print_bar()
		utils.save_results_on_file(ARGS.data_folder, Q_list, A_list, results)

	if ARGS.plot:
		utils.print_bar()
		results_folder = os.path.join(ARGS.data_folder, 'tf_results')
		run_folder = generate_results(results_folder, Q_list, A_list, results, sample=ARGS.sample, save_thresh=ARGS.save_thresh, test=ARGS.test_mode)

		if ARGS.memo:
			fname = os.path.join(run_folder, 'description.txt')
			with open(fname, 'w+') as file:
				file.write(ARGS.memo + '\n')

	utils.checkpoint()
	utils.print_BAR()
	print()
	if ARGS.memo:
		print(ARGS.memo)
		utils.print_BAR()
	print()

	return partitions, Q_list, A_list, evidence_list, results



parser = argparse.ArgumentParser(description='Evaluate using P&D style question-generation and -answering')
parser.add_argument('news_gen_file', help='Path to file used for partition into Question set and Answer set')
parser.add_argument('data_folder', help='Path to data folder including freebase entity types and predicate substitution pairs')
parser.add_argument('--uu-graphs', help='Path to Univalent entailment graphs to assist question answering')
parser.add_argument('--bu-graphs', help='Path to Bivalent entailment graphs to assist question answering')
parser.add_argument('--sim-cache', action='store_true', help='Use a similarity cache to answer questions (file must be located in data folder)')
parser.add_argument('--bv-emb', help='Path to embeddings of the Bivalent graphs used to augment question answering')
parser.add_argument('--uv-emb', help='Path to embeddings of the Univalent graphs used to augment question answering')
# parser.add_argument('--filter-qs', action='store_true', help='Use PPDB to filter questions during generation (file must be located in data folder)')
# parser.add_argument('--test-all', action='store_true', help='Test all variations of the given configuration')
parser.add_argument('--ppdb', help='Path to PPDB to answer questions')
parser.add_argument('--text-EGs', action='store_true', help='Read in plain-text entailment graphs from a folder')
parser.add_argument('--local', action='store_true', help='Read in local entailment graphs (default is global)')
parser.add_argument('--eval-fun', default='acc', help='Evaluate results using the specified test function')
# parser.add_argument('--day-range', type=int, default=0, help='Evidence window around the day questions are drawn from (extends # days in both directions)')
parser.add_argument('--graph-qs', action='store_true', help='Ask only questions which have hits in the graph')
parser.add_argument('--backoff', default='none', choices=['none', 'node', 'edge'], help='Back off to wrong-type graphs if right-type graphs can\'t answer')

parser.add_argument('--test-mode', action='store_true', help='Use test set data (default is dev set)')

parser.add_argument('--plot', action='store_true', help='Plot results visually')
parser.add_argument('--sample', action='store_true', help='Sample results at specified precision cutoff')

parser.add_argument('--save-results', action='store_true', help='')
parser.add_argument('--save-thresh', action='store_true', help='Save precision-level cutoffs for entailment graphs')
parser.add_argument('--save-qs', action='store_true', help='Save generated questions to file')
parser.add_argument('--save-preds', action='store_true', help='Save top predicates to file')
parser.add_argument('--save-errors', action='store_true', help='Save erroneous propositions to file')

parser.add_argument('--no-answer', action='store_true', help='Stop execution before answering questions')
parser.add_argument('--quick', action='store_true', help='Skip long-running optional work (e.g. computing B->U)')

parser.add_argument('--memo', help='Description of the experiment to be recorded with results')

if __name__ == '__main__':
	partitions, Q, A, E, results = run_evaluate()
