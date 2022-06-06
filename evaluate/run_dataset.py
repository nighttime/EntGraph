import argparse
import json
import statistics
from argparse import Namespace
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

import proposition
import reference
import utils
from analyze_dataset import analyze_results, make_results_folder
from answer_tf import _make_b_arg_cache, _make_prop_cache, infer_claim_BB, answer_tf_sets
import run_mv
from graph_encoder import GraphDeducer
from proposition import *
from entailment import *
from answer_wh import *

from lemma_baseline import baseline

from typing import *


# Program Hyperparameters

ARGS = None

def eval_dataset(Ent_list: List[Tuple[Prop, Prop]], models: Dict[str, Any], answer_modes: Set[str], A_list: List[int]) -> Tuple[List[float], Dict[str, Any]]:
	global ARGS
	answers = []
	nearest_found_edge = 0
	nearest_no_found_edge = 0

	sim_scores = []
	dataset_log = []

	no_lhs_ct = no_rhs_ct = no_lhs_and_rhs_ct = 0

	for i,ent in enumerate(Ent_list):
		log = {'LM Backoff': 0}
		if i % 500 == 0:
			utils.print_progress(i/len(Ent_list), 'question {}'.format(i))

		score = 0

		lemma_true = False
		if 'Lemma Baseline' in answer_modes:
			score = models['Lemma Baseline'][i]
			lemma_true = score == 1

		if ent is None:
			answers.append(score)
			dataset_log.append(log)
			continue

		## Predictions #################

		lhs, rhs = ent

		if 'Literal B' in answer_modes:
			if lhs.prop_desc() == rhs.prop_desc():
				score = 1

		if 'BB' in answer_modes:
			if not ('BB-LM' in answer_modes and ARGS.ablate != 'none'):
				# Create a prop-indexed fact cache of A: {pred_desc : [prop]} for exact-match and EG lookup
				# prop_facts_b = _make_prop_cache([lhs], removals=[rhs])
				prop_facts_b = _make_prop_cache([lhs], removals=[rhs])
				# Create an arg-indexed fact cache of A: {arg : [(prop, arg_idx)]} for similarity lookup
				arg_facts_b = _make_b_arg_cache([lhs], removals=[rhs])

				bb_score, bb_support = infer_claim_BB(rhs, prop_facts_b, arg_facts_b, models['BU'])
				if bb_support:
					score = max(score, bb_score)

			q_typing = '#'.join(lhs.types)
			lhs_in_graph = (q_typing in models['BU'] and lhs.pred_desc() in models['BU'][q_typing].nodes)
			rhs_in_graph = (q_typing in models['BU'] and rhs.pred_desc() in models['BU'][q_typing].nodes)

			if score == 0:
				if not lhs_in_graph and not rhs_in_graph:
					no_lhs_and_rhs_ct += 1
				elif not lhs_in_graph:
					no_lhs_ct += 1
				elif not rhs_in_graph:
					no_rhs_ct += 1


			# if 'BB-LM' in answer_modes and score == 0:
			# if 'BB-LM' in answer_modes and not (lhs_in_graph and rhs_in_graph) and score == 0:
			# if 'BB-LM' in answer_modes and rhs_in_graph and not lhs_in_graph and score == 0:
			# if 'BB-LM' in answer_modes and lhs_in_graph and not rhs_in_graph and score == 0:
			# if 'BB-LM' in answer_modes and (rhs_in_graph != lhs_in_graph) and score == 0:
			# if 'BB-LM' in answer_modes and score == 0 and ((not lhs_in_graph and rhs_in_graph) or (rhs_in_graph and lhs_in_graph and ARGS.ablate != 'none')):
			if 'BB-LM' in answer_modes and score == 0 and ((lhs_in_graph and not rhs_in_graph) or (rhs_in_graph and lhs_in_graph and ARGS.ablate != 'none')):
				log['LM Backoff'] = 1
				log['lhs-prime'] = log['rhs-prime'] = log['lhs weight'] = log['rhs weight'] = None

				# if A_list[i] == 0 and q_typing in models['BU'] and len(models['BU'][q_typing].edges) < 50:
				# 	answers.append(score)
				# 	dataset_log.append(log)
				# 	continue

				k = 4
				# k = 'logscale'
				# k = 'proportional'

				ablated_preds = []
				if ARGS.ablate in ['p', 'pq']:
					ablated_preds.append(lhs.pred_desc())
				if ARGS.ablate in ['q', 'pq']:
					ablated_preds.append(rhs.pred_desc())

				default_res_l = ([lhs.pred_desc()], [1.0])
				default_res_r = ([rhs.pred_desc()], [1.0])

				if not lhs_in_graph or ARGS.ablate in ['p', 'pq']:
					res_l = models['BB-Deducer'].get_nearest_node(lhs.pred_desc(), k=k, position='left', model=models['BU'], ablated_preds=ablated_preds)
					res_l = res_l or default_res_l
				else:
					res_l = default_res_l

				if not rhs_in_graph or ARGS.ablate in ['q', 'pq']:
					res_r = models['BB-Deducer'].get_nearest_node(rhs.pred_desc(), k=k, position='right', model=models['BU'], ablated_preds=ablated_preds)
					res_r = res_r or default_res_r
				else:
					res_r = default_res_r

				# res_l = default_res_l if lhs_in_graph else (res_l or default_res_l)
				# res_r = default_res_r if rhs_in_graph else (res_r or default_res_r)

				prev_score = score
				for lhs_pred, lhs_score in zip(*res_l):
					lhs = Prop.with_new_pred(lhs, lhs_pred)
					if lhs_pred != ent[0]:
						sim_scores.append(lhs_score)
					for rhs_pred, rhs_score in zip(*res_r):
						rhs = Prop.with_new_pred(rhs, rhs_pred)
						# if (lhs, rhs) != ent:
						# if rhs_pred != ent[1]:
						# 	sim_scores.append(rhs_score)

						# Create a prop-indexed fact cache of A: {pred_desc : [prop]} for exact-match and EG lookup
						prop_facts_b = _make_prop_cache([lhs], removals=[rhs])
						# Create an arg-indexed fact cache of A: {arg : [(prop, arg_idx)]} for similarity lookup
						arg_facts_b = _make_b_arg_cache([lhs], removals=[rhs])

						bb_score, bb_support = infer_claim_BB(rhs, prop_facts_b, arg_facts_b, models['BU'])
						if bb_support:
							possible_score = bb_score * lhs_score * rhs_score
							# possible_score = (bb_score * min(lhs_score, rhs_score))
							if possible_score > score:
								score = possible_score
								log['lhs-prime'] = lhs.pred_desc()
								log['rhs-prime'] = rhs.pred_desc()
								log['lhs weight'] = lhs_score
								log['rhs weight'] = rhs_score

				if score > prev_score:
					nearest_found_edge += 1
				else:
					nearest_no_found_edge += 1

		answers.append(score)
		log['score'] = score
		dataset_log.append(log)

	utils.print_progress(1, 'done')

	if 'BB-LM' in answer_modes:
		if len(models['BB-Deducer'].log) > 0:
			print('Deducer log:')
			print(models['BB-Deducer'].log)

		print('Times nearest nodes lead to an edge: {}/{}'.format(nearest_found_edge, (nearest_found_edge+nearest_no_found_edge)))

	if sim_scores:
		print('Sim score stats:')
		print('min', min(sim_scores))
		print('max', max(sim_scores))
		print('mean', statistics.mean(sim_scores))
		print('stddev', statistics.stdev(sim_scores))

	print('no lhs & no rhs: {}'.format(no_lhs_and_rhs_ct))
	print('no lhs only: {}'.format(no_lhs_ct))
	print('no rhs only: {}'.format(no_rhs_ct))

	return answers, dataset_log

def eval_lemma_baseline(data_folder: str, q_mask: List[int], args) -> List[float]:
	fname = 'test.txt' if args.test_mode else 'dev.txt'
	dataset_path = os.path.join(data_folder, 'levy_holt', fname)
	lemma = baseline.predict_lemma_baseline(dataset_path, Namespace(dev_sherliic_v2=False, test_sherliic_v2=False))
	result = lemma[q_mask].astype(float).tolist()
	return result


def run_evaluate():
	global ARGS
	ARGS = parser.parse_args()

	embedding_deduction = False
	if ARGS.graph_embs:
		if not ARGS.model:
			print('Both graph embeddings and an online model must be specified to use graph embedding deduction')
			exit(1)
		else:
			embedding_deduction = True

	print()
	utils.print_BAR()
	mode = 'local' if reference.RUNNING_LOCAL else 'server'
	title = 'Running Eval: {} ({})'.format(ARGS.dataset, mode)
	if ARGS.quick:
		title += ' (quick mode)'
	print(title)

	if ARGS.test_mode:
		print('* EVAL on TEST SET')

	original_types_file = 'data/freebase_types/entity2Types.txt'
	print('* Using data folder: ' + ARGS.data_folder)

	resources = ['lemma baseline']
	# if ARGS.uu_graphs is not None:
	# 	resources.append('U->U graphs')
	if ARGS.bu_graphs is not None:
		resources.append('B->B/U graphs')
	if ARGS.sim_cache:
		resources.append('Similarity cache NYI')
	if ARGS.ppdb:
		resources.append('PPDB NYI')
	print('* Using evidence from:', str(resources))

	question_modes = set()
	question_modes |= {'binary'}

	print('* Question modes:', question_modes)

	answer_modes = set()
	if 'binary' in question_modes:
		answer_modes.add('BB')

	if ARGS.backoff != 'none':
		reference.GRAPH_BACKOFF = ARGS.backoff
		print('* Answer modes:', answer_modes, '[+Backoff={}]'.format(ARGS.backoff))
	else:
		print('* Answer modes:', answer_modes)

	if embedding_deduction:
		reference.SMOOTHING_SELF_LOOPS = not ARGS.smoothing_no_self
		reference.TRIGGER = reference.TRIGGER(ARGS.smoothing_trigger)
		print(f'* EG-LM KNN smoothing with: {ARGS.model}{", self-loops" if not ARGS.smoothing_no_self else ""}, trigger={reference.TRIGGER.value}')
		reference.SMOOTHING_K = ARGS.smoothing_K
		reference.SMOOTH_P = reference.SMOOTH(ARGS.smooth_P)
		reference.SMOOTH_Q = reference.SMOOTH(ARGS.smooth_Q)
		print(f'* Smoothing K: {reference.SMOOTHING_K} P: {reference.SMOOTH_P.value} Q: {reference.SMOOTH_Q.value}')

	utils.checkpoint()
	utils.print_BAR()

	#######################################

	models = {}

	# print('Reading graphs from: {}'.format(ARGS.bu_graphs))
	bu_graphs = load_graphs(ARGS.bu_graphs, 'Reading B->B/U graphs...', ARGS)
	if bu_graphs:
		models['BU'] = bu_graphs
	print('Read {} graphs'.format(len(bu_graphs)))
	utils.checkpoint()
	utils.print_BAR()

	if ARGS.graph_embs:
		print('Using graph embeddings from: {}'.format(ARGS.graph_embs))
		models['BV-B-Deducer'] = GraphDeducer(ARGS.model, ARGS.graph_embs)

	if ARGS.P_wordnet:
		if ARGS.P_wn_relation:
			reference.P_WN_RELATION = reference.WN_RELATION(ARGS.P_wn_relation)
		print(f'Using WordNet {reference.P_WN_RELATION} suggestions for P-smoothing from: {ARGS.P_wordnet}')
		with open(ARGS.P_wordnet) as f:
			models['WN-P'] = json.load(f)

	if ARGS.Q_wordnet:
		if ARGS.Q_wn_relation:
			reference.Q_WN_RELATION = reference.WN_RELATION(ARGS.Q_wn_relation)
		print(f'Using WordNet {reference.Q_WN_RELATION} suggestions for Q-smoothing from: {ARGS.Q_wordnet}')
		with open(ARGS.Q_wordnet) as f:
			models['WN-Q'] = json.load(f)

	if ARGS.PQ_hyp_tree and (ARGS.P_hyper_tree or ARGS.Q_hypo_tree):
		print(f'Using precomputed hypernymy tree for PQ smoothing: {ARGS.PQ_hyp_tree}')
		with open(ARGS.PQ_hyp_tree, 'rb') as f:
			graph_trees: Dict = pickle.load(f)
			rev_typing = {'#'.join(reversed(typing.split('#'))):v for typing,v in graph_trees.items() if len(set(typing.split('#'))) == 2}
			graph_trees.update(rev_typing)
			if ARGS.P_hyper_tree:
				models['hyp-tree-P'] = graph_trees
			if ARGS.Q_hypo_tree:
				models['hyp-tree-Q'] = graph_trees

	print('Reading in dataset...')
	Ent_list, answers, q_idx = utils.read_dataset(ARGS.dataset, ARGS.data_folder, test=ARGS.test_mode, directional=ARGS.directional)
	print('Dataset: {} Questions'.format(len(Ent_list)))
	utils.checkpoint()

	if ARGS.save_dataset_preds:
		fname = f'dataset_preds_{ARGS.dataset}.txt'
		with open(fname, 'w+') as file:
			for p in list(set([e for pair in Ent_list for e in pair or []])):
				base_term = proposition.extract_predicate_base_term(p.pred)
				file.write(f'{base_term}\n')
		print(f'Saved dataset preds to file: {fname}')

	if ARGS.no_answer:
		print('Finishing before answering phase.')
		return

	# Answer the questions using available resources: A set, U->U Graph, B->U graph
	utils.print_BAR()
	print('Predicting answers...')

	results = {}
	basic_answers = set()
	if ARGS.dataset == 'levy_holt' and not ARGS.directional:
		print('Lemma Baseline')
		lemma = eval_lemma_baseline(os.path.join(ARGS.data_folder, 'datasets'), q_idx, ARGS)
		results['*Lemma Baseline'] = ([[l] for l in lemma], None)
		models['Lemma Baseline'] = lemma
		basic_answers.add('Lemma Baseline')

	# if 'binary' in question_modes:
	# 	# answer_modes = {'Literal B'}
	# 	answer_modes = {'Lemma Baseline'}
	# 	results['*Lemma Baseline'] = eval_dataset(Ent_list, models, answer_modes)
	Q_List, A_List, evidence_List = [], [], []
	if bu_graphs and 'binary' in question_modes:
		print('BB Baseline')
		# answer_modes = {'BB', 'Literal B'}
		answer_modes = basic_answers | {'BB'}
		# results['BB'] = eval_dataset(Ent_list, models, answer_modes, A_list)[0]

		Q_List = [[ent[1]] if ent else [] for ent in Ent_list]
		evidence_List = [([],[ent[0]]) if ent else ([],[]) for ent in Ent_list]
		A_List = [[a] for a in answers]
		results['BB'] = answer_tf_sets(Q_List, evidence_List, models, answer_modes)[:2]

		if ARGS.graph_embs:
			print('LM Nearest')
			answer_modes = basic_answers | {'BB', 'BB-LM'}
			# results['BB-LM'], log = eval_dataset(Ent_list, models, answer_modes, A_list)
			results['BB-LM'] = answer_tf_sets(Q_List, evidence_List, models, answer_modes)[:2]

	reference.FINISH_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')
	results_folder = os.path.join(ARGS.data_folder, '{}_results'.format(ARGS.dataset))
	run_folder = make_results_folder(results_folder, test=ARGS.test_mode)

	if ARGS.memo:
		fname = os.path.join(run_folder, 'description.txt')
		with open(fname, 'w+') as file:
			file.write(f"{ARGS.dataset} {'(directional)' if ARGS.directional else ''}\n")
			file.write(ARGS.bu_graphs + '\n\n')
			for key in vars(ARGS):
				if any(key.startswith(prefix) for prefix in ['P', 'Q', 'smooth']):
					file.write(f'{key}:\t{getattr(ARGS, key)}\n')

			if ARGS.memo:
				file.write(f'\nmemo: {ARGS.memo}\n')

			file.write('\n\n\n' + ' '.join(sys.argv))

	if ARGS.save_results and 'BB-LM' in results:
		utils.print_bar()
		# utils.save_results_on_file(ARGS.data_folder, Q_List, A_List, results, memo=ARGS.memo)
		ds_name = reference.DS_SHORT_NAME_TO_DISPLAY[ARGS.dataset] + (' (Directional)' if ARGS.directional else '')
		utils.save_results_on_file(ARGS.data_folder, Ent_list, answers, results, memo=ARGS.memo, name=ds_name)

	if ARGS.sample and 'BB-LM' in results:
		assert len(results['BB-LM'][0]) == len(Ent_list)
		log_fname = os.path.join(run_folder, 'results_log.txt')
		with open(log_fname, 'w+') as f:
			f.write('Typing\tGraph Sz\tScore\tTruth\tLHS\'\tLHS weight\tRHS\'\tRHS weight\tLHS\tRHS\n')
			for i, ent in enumerate(Ent_list):
				if ent is None:
					f.write('[parse fail]\t-\t-\t-\t-\t-\t-\t-\t-\t-\n')
					continue
				typing = '#'.join(ent[0].basic_types)
				graph_size = len(models['BU'][typing].edges.keys()) if typing in models['BU'] else 0
				score_code = -1 if results['*Lemma Baseline'][0][i] else (-2 if results['BB'][0][i] else results['BB-LM'][0][i])
				lhs_pred = rhs_pred = lhs_weight = rhs_weight = None
				q_log = results['BB-LM'][1][i][0]
				if 'LM Backoff' in q_log and q_log['LM Backoff']:
					lhs_pred = log[i]['lhs-prime']
					rhs_pred = log[i]['rhs-prime']
					lhs_weight = log[i]['lhs weight']
					rhs_weight = log[i]['rhs weight']
				f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(typing, graph_size, score_code, A_List[i][0], lhs_pred, lhs_weight, rhs_pred, rhs_weight, ent[0].pred_desc(), ent[1].pred_desc()))
			utils.print_bar()
			print('Results log written to {}'.format(log_fname))

	if ARGS.plot:
		utils.print_bar()
		if not os.path.exists(results_folder):
			os.makedirs(results_folder)
		analyze_results(run_folder, Ent_list, A_List, results, dataset=ARGS.dataset, test=ARGS.test_mode, directional=ARGS.directional)

	utils.checkpoint()
	utils.print_BAR()
	if ARGS.memo:
		print(ARGS.memo)
		utils.print_BAR()




parser = argparse.ArgumentParser(description='Evaluate using a provided dataset')
parser.add_argument('data_folder', help='Path to data folder including freebase entity types and predicate substitution pairs')
parser.add_argument('--dataset', required=True, default='levy_holt', choices=['levy_holt', 'sl_ant'], help='Dataset name to evaluate on')
parser.add_argument('--uu-graphs', help='Path to Unary->Unary entailment graphs to assist question answering')
parser.add_argument('--bu-graphs', help='Path to Binary->Unary entailment graphs to assist question answering')
parser.add_argument('--sim-cache', action='store_true', help='Use a similarity cache to answer questions (file must be located in data folder)')
parser.add_argument('--ppdb', help='Path to PPDB to answer questions')
parser.add_argument('--text-EGs', action='store_true', help='Read in plain-text entailment graphs from a folder')
parser.add_argument('--local', action='store_true', help='Read in local entailment graphs (default is global)')
parser.add_argument('--eval-fun', default='acc', help='Evaluate results using the specified test function')
parser.add_argument('--graph-qs', action='store_true', help='Ask only questions which have hits in the graph')
parser.add_argument('--backoff', default='none', choices=['none', 'node', 'edge', 'both_nodes'], help='Back off to wrong-type graphs if right-type graphs can\'t answer')

parser.add_argument('--graph-embs', help='Path to folder containing LM-encoded graph nodes')
parser.add_argument('--model', choices=['bert', 'roberta'], default='roberta', help='Choice of model used to encode graph predicates')

parser.add_argument('--test-mode', action='store_true', help='Use test set data (default is dev set)')
parser.add_argument('--directional', action='store_true', help='Use only the directional subset of the dataset, if available')
# parser.add_argument('--ablate', default='none', choices=['none', 'p', 'q', 'pq'], help='Ablate the p |= q relation by deleting a node(s) from the entailment graph')


parser.add_argument('--smoothing-trigger', default=reference.TRIGGER.NO_EG_SUPPORT.value, choices=[e.value for e in reference.TRIGGER], help='Activate smoothing according to a condition')
parser.add_argument('--smoothing-no-self', action='store_true', help='Do not predict the smoothing target as its own smoothing prediction')

parser.add_argument('--smoothing-K', default=4, type=int, help='LM smoothing using KNN in EG: sets K param')
parser.add_argument('--smooth-P', default=reference.SMOOTH.NONE.value, choices=[e.value for e in reference.SMOOTH], help='Activate smoothing on the left-hand side of an entailment query e.g. P |= Q')
parser.add_argument('--smooth-Q', default=reference.SMOOTH.NONE.value, choices=[e.value for e in reference.SMOOTH], help='Activate smoothing on the right-hand side of an entailment query e.g. P |= Q')

parser.add_argument('--P-wordnet', help='Smooth P using hyponym/hypernym possibilities from WordNet')
parser.add_argument('--Q-wordnet', help='Smooth Q using hyponym/hypernym possibilities from WordNet')
parser.add_argument('--P-wn-relation', default=reference.WN_RELATION.HYPERNYM.value, choices=[e.value for e in reference.WN_RELATION], help='Smooth P using the given relation from WordNet')
parser.add_argument('--Q-wn-relation', default=reference.WN_RELATION.HYPONYM.value, choices=[e.value for e in reference.WN_RELATION], help='Smooth P using the given relation from WordNet')

parser.add_argument('--PQ-hyp-tree', help='Smooth P and Q using precomputed hypernymy tree')
parser.add_argument('--P-hyper-tree', action='store_true', help='Smooth P using hypernym pseudo-tree')
parser.add_argument('--Q-hypo-tree', action='store_true', help='Smooth Q using hyponym pseudo-tree')

parser.add_argument('--plot', action='store_true', help='Plot results visually')

parser.add_argument('--sample', action='store_true', help='Sample results at specified precision cutoff')

parser.add_argument('--save-results', action='store_true', help='')
parser.add_argument('--save-dataset-preds', action='store_true', help='Store a list of the unique dataset predicate base terms')
# parser.add_argument('--save-thresh', action='store_true', help='Save precision-level cutoffs for entailment graphs')
# parser.add_argument('--save-qs', action='store_true', help='Save generated questions to file')
# parser.add_argument('--save-preds', action='store_true', help='Save top predicates to file')
# parser.add_argument('--save-errors', action='store_true', help='Save erroneous propositions to file')

parser.add_argument('--no-answer', action='store_true', help='Stop execution before answering questions')
parser.add_argument('--quick', action='store_true', help='Skip long-running optional work (e.g. computing B->U)')

parser.add_argument('--memo', help='Description of the experiment to be recorded with results')

if __name__ == '__main__':
	run_evaluate()
