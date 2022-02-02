import argparse
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

import reference
import utils
from analyze_dataset import analyze_results, make_results_folder
from answer_tf import _make_b_arg_cache, _make_prop_cache, infer_claim_BB
from graph_encoder import GraphDeducer
from proposition import *
from entailment import *
from answer_wh import *

from lemma_baseline import baseline

from typing import *


# Program Hyperparameters

ARGS = None

# (treatment.for.1,treatment.for.2) vancomycin::medicine infection::disease
def read_lh_prop(prop_str: str) -> Prop:
	bare_pred, typed_arg1, typed_arg2 = prop_str.split()
	types = [a.split('::')[1] for a in [typed_arg1, typed_arg2]]
	if types[0] == types[1]:
		types = [types[0] + '_1', types[1] + '_2']
		typed_arg1, typed_arg2 = typed_arg1 + '_1', typed_arg2 + '_2'
	pred_desc = '#'.join([bare_pred, *types])
	arg_desc = [a.replace('::', '#') for a in [typed_arg1, typed_arg2]]
	prop = Prop.from_descriptions(pred_desc, arg_desc)
	prop.set_entity_types('EE')
	return prop

def read_lh(path: str, directional=False) -> Tuple[List[Optional[Tuple[Prop, Prop]]], List[int], List[int]]:
	print('Reading {}'.format(path))
	Ent_list, A_list = [], []
	issue_ct = 0
	with open(path) as file:
		for line in file:
			parts = line.strip().split('\t')
			answer = 1 if parts[-1] == 'True' else 0

			if len(parts) != 3 or not all(parts):
				issue_ct += 1
				Ent_list.append(None)
				A_list.append(answer)
				continue

			rel_str1, rel_str2, _ = parts
			prop1, prop2 = read_lh_prop(rel_str1), read_lh_prop(rel_str2)
			Ent_list.append((prop2, prop1))
			A_list.append(answer)

	q_mask = list(range(len(Ent_list)))
	if directional:
		all_qa = {Ent_list[i]:A_list[i] for i in range(len(Ent_list))}
		dir_qs = []
		reverses = set()
		q_mask = []
		for i,q in enumerate(Ent_list):
			if not q:
				continue
			if q in reverses:
				q_mask.append(i)
				dir_qs.append((q, all_qa[q]))
				continue
			rev_q = tuple(reversed(q))
			if rev_q in all_qa and all_qa[q] != all_qa[rev_q]:
				q_mask.append(i)
				dir_qs.append((q, all_qa[q]))
				reverses.add(rev_q)
		new_Ent_list, new_A_list = tuple(zip(*dir_qs))
		Ent_list, A_list = new_Ent_list, new_A_list

	assert len(q_mask) == len(Ent_list)

	return Ent_list, A_list, q_mask

def read_dataset(dataset: str, test=False, directional=False) -> Tuple[List[Tuple[Prop, Prop]], List[int], List[int]]:
	if dataset == 'levy_holt':
		fname = '{}_rels.txt'.format('test' if test else 'dev')
		path = os.path.join(ARGS.data_folder, 'levy_holt', fname)
		return read_lh(path, directional=directional)

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
		print('* Embedding deduction: {}'.format(ARGS.model))

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
		models['BB-Deducer'] = GraphDeducer(ARGS.model, ARGS.graph_embs)

	print('Reading in dataset...')
	Ent_list, A_list, q_mask = read_dataset(ARGS.dataset, test=ARGS.test_mode, directional=ARGS.lh_directional)
	print('Dataset: {} Questions'.format(len(Ent_list)))

	utils.checkpoint()

	# Answer the questions using available resources: A set, U->U Graph, B->U graph
	utils.print_BAR()
	print('Predicting answers...')

	results = {}

	if ARGS.dataset == 'levy_holt':
		print('Lemma Baseline')
		lemma = eval_lemma_baseline(ARGS.data_folder, q_mask, ARGS)
		results['*Lemma Baseline'] = lemma
		models['Lemma Baseline'] = lemma

	# if 'binary' in question_modes:
	# 	# answer_modes = {'Literal B'}
	# 	answer_modes = {'Lemma Baseline'}
	# 	results['*Lemma Baseline'] = eval_dataset(Ent_list, models, answer_modes)
	log = []
	if bu_graphs and 'binary' in question_modes:
		print('BB Baseline')
		# answer_modes = {'BB', 'Literal B'}
		answer_modes = {'BB', 'Lemma Baseline'}
		results['BB'] = eval_dataset(Ent_list, models, answer_modes, A_list)[0]

		if ARGS.graph_embs:
			print('LM Nearest')
			answer_modes.add('BB-LM')
			results['BB-LM'], log = eval_dataset(Ent_list, models, answer_modes, A_list)

	reference.FINISH_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')
	results_folder = os.path.join(ARGS.data_folder, '{}_results'.format(ARGS.dataset))
	run_folder = make_results_folder(results_folder, test=ARGS.test_mode)

	if ARGS.memo:
		fname = os.path.join(run_folder, 'description.txt')
		with open(fname, 'w+') as file:
			file.write(ARGS.memo + '\n')

	if ARGS.save_results and 'BB-LM' in results:
		assert len(log) == len(Ent_list)
		log_fname = os.path.join(run_folder, 'results_log.txt')
		with open(log_fname, 'w+') as f:
			f.write('Typing\tGraph Sz\tScore\tTruth\tLHS\'\tLHS weight\tRHS\'\tRHS weight\tLHS\tRHS\n')
			for i, ent in enumerate(Ent_list):
				if ent is None:
					f.write('[parse fail]\t-\t-\t-\t-\t-\t-\t-\t-\t-\n')
					continue
				typing = '#'.join(ent[0].basic_types)
				graph_size = len(models['BU'][typing].edges.keys()) if typing in models['BU'] else 0
				score_code = -1 if results['*Lemma Baseline'][i] else (-2 if results['BB'][i] else results['BB-LM'][i])
				lhs_pred = rhs_pred = lhs_weight = rhs_weight = None
				if log[i]['LM Backoff']:
					lhs_pred = log[i]['lhs-prime']
					rhs_pred = log[i]['rhs-prime']
					lhs_weight = log[i]['lhs weight']
					rhs_weight = log[i]['rhs weight']
				f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(typing, graph_size, score_code, A_list[i], lhs_pred, lhs_weight, rhs_pred, rhs_weight, ent[0].pred_desc(), ent[1].pred_desc()))
			utils.print_bar()
			print('Results log written to {}'.format(log_fname))

	if ARGS.plot:
		utils.print_bar()
		if not os.path.exists(results_folder):
			os.makedirs(results_folder)
		analyze_results(run_folder, Ent_list, A_list, results, test=ARGS.test_mode, directional=ARGS.lh_directional)

	utils.checkpoint()
	utils.print_BAR()
	if ARGS.memo:
		print(ARGS.memo)
		utils.print_BAR()
	print()




parser = argparse.ArgumentParser(description='Evaluate using a provided dataset')
parser.add_argument('data_folder', help='Path to data folder including freebase entity types and predicate substitution pairs')
parser.add_argument('dataset', default='levy_holt', choices=['levy_holt'], help='Dataset name to evaluate on')
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
parser.add_argument('--lh-directional', action='store_true', help='Use only the directional subset of L/H')
parser.add_argument('--ablate', default='none', choices=['none', 'p', 'q', 'pq'], help='Ablate the p |= q relation by deleting a node(s) from the entailment graph')

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
	run_evaluate()
