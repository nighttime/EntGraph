import argparse
import sys
import os
import re
from collections import defaultdict, Counter, namedtuple
from operator import itemgetter, attrgetter
import itertools
import random
from datetime import datetime
import numpy as np

ScoredSupport = namedtuple('ScoredSupport', ['score', 'prop'])

from proposition import *
from entailment import *
from article import *
from analyze import *
from questions import *
from answer_wh import *
from answer_tf import *

from typing import *


# For printing
BAR_LEN = 50
BAR = '=' * BAR_LEN
bar = '-' * BAR_LEN

# Program Hyperparameters
# Fraction of props partitioned into Q
QA_SPLIT = 0.5
TOP_K_ENTS = 100

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
	print(bar)
	predictions = answer_questions(questions, evidence, uu_graphs=uu_graphs, bu_graphs=bu_graphs)
	score = eval_fun(answers, predictions)
	print(score_text.format(score))
	checkpoint()


def run_wh_sets(Q_list: List[List[str]], A_list: List[List[Set[str]]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
				score_text: str, eval_fun: Callable[[List[Set[str]], List[List[str]]], float],
				uu_graphs: EGraphCache=None, bu_graphs: EGraphCache=None):
	print(bar)
	predictions_list, _ = answer_question_sets(Q_list, evidence_list, uu_graphs=uu_graphs, bu_graphs=bu_graphs, A_list=A_list)
	score = eval_wh_sets(A_list, predictions_list, eval_fun=eval_fun)
	print(score_text)
	print('{:.4f}'.format(score))
	checkpoint()

def run_tf_sets(Q_list: List[List[Prop]], A_list: List[List[int]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
				score_text: str, eval_fun: Callable[[List[Any], List[Any]], float],
				uu_graphs: EGraphCache=None, bu_graphs: EGraphCache=None) -> List[List[float]]:
	print(bar)
	print(score_text)
	scores_list = answer_tf_sets(Q_list, evidence_list, uu_graphs=uu_graphs, bu_graphs=bu_graphs)
	predictions_list = threshold_sets(scores_list, 0)
	p = eval_tf_sets(A_list, predictions_list, eval_fun=precision)
	r = eval_tf_sets(A_list, predictions_list, eval_fun=recall)
	a = eval_tf_sets(A_list, predictions_list, eval_fun=acc)
	print('precision: {:.3f}'.format(p))
	print('recall:    {:.3f}'.format(r))
	print('accuracy:  {:.3f}'.format(a))
	checkpoint()

	return scores_list


def checkpoint():
	print('+ Checkpoint:', datetime.now().strftime('%H:%M:%S'))


def main():
	global ARGS
	ARGS = parser.parse_args()

	eval_fun = globals()[ARGS.eval_fun]

	print()
	print(BAR)
	print('Running Eval')

	original_types_file = 'data/freebase_types/entity2Types.txt'
	print('* Using entity types from: ' + ARGS.entity_types)

	resources = ['literal answers']
	if ARGS.uu_graphs is not None:
		resources.append('U->U graphs')
	if ARGS.bu_graphs is not None:
		resources.append('B->U graphs')
	print('* Using evidence from:', str(resources))

	# print('* Evaluating with: ' + eval_fun.__name__)

	checkpoint()
	print(BAR)

	#######################################

	print('Loading entity type cache...')
	if not os.path.exists(ARGS.entity_types):
		print('  !generating new cache...')
		load_entity_types(original_types_file)
		save_precomputed_entity_types(ARGS.entity_types)
	load_precomputed_entity_types(ARGS.entity_types)

	# Reading in news articles
	articles, unary_props, binary_props = read_source_data(ARGS.news_gen_file)

	# Generate questions from Q
	print('Generating questions...', end=' ', flush=True)
	P_list, N_list, evidence_list = generate_tf_question_sets(articles)
	num_Qs = sum(len(qs) for qs in P_list + N_list)
	num_P_Qs = sum(len(qs) for qs in P_list)
	num_sets = len(P_list)
	print('Generated {} questions ({:.1f}% +) from {} sets'.format(num_Qs, (num_P_Qs/num_Qs)*100, num_sets))

	Q_list, A_list = format_tf_QA_sets(P_list, N_list)

	uu_graphs, bu_graphs = None, None
	graph_ext = 'sim' if ARGS.local else 'binc'
	# Open U->U entailment graphs
	if ARGS.uu_graphs:
		print('Reading U->U graphs...', end=' ', flush=True)
		if ARGS.raw_EGs:
			uu_graphs = read_graphs(ARGS.uu_graphs, ext=graph_ext)
		else:
			uu_graphs = load_precomputed_EGs(ARGS.uu_graphs)
		print('Graphs read: {}'.format(len(uu_graphs)))

	# Open B->U entailment graphs
	if ARGS.bu_graphs:
		print('Reading B->U graphs...', end=' ', flush=True)
		if ARGS.raw_EGs:
			bu_graphs = read_graphs(ARGS.bu_graphs, ext=graph_ext)
		else:
			bu_graphs = load_precomputed_EGs(ARGS.bu_graphs)
		print('Graphs read: {}'.format(len(bu_graphs)))

	checkpoint()
	print(BAR)

	# Answer the questions using available resources: A set, U->U Graph, B->U graph
	print('Predicting answers...')

	results = {}

	if ARGS.test_all:
		r = run_tf_sets(Q_list, A_list, evidence_list, 'form-dependent baseline', eval_fun=eval_fun)
		results['baseline'] = r

		if uu_graphs:
			r = run_tf_sets(Q_list, A_list, evidence_list, 'With U->U', eval_fun=eval_fun, uu_graphs=uu_graphs)
			results['U->U'] = r

		if bu_graphs:
			r = run_tf_sets(Q_list, A_list, evidence_list, 'With B->U', eval_fun=eval_fun, bu_graphs=bu_graphs)
			results['B->U'] = r

		if uu_graphs and bu_graphs:
			r = run_tf_sets(Q_list, A_list, evidence_list, 'With U->U and B->U', eval_fun=eval_fun,
						uu_graphs=uu_graphs, bu_graphs=bu_graphs)
			results['U->U and B->U'] = r

	else:
		run_tf_sets(Q_list, A_list, evidence_list, 'score', eval_fun=eval_fun, uu_graphs=uu_graphs, bu_graphs=bu_graphs)

	if ARGS.plot:
		plot_results(A_list, results)

	print(BAR)
	print()



parser = argparse.ArgumentParser(description='Evaluate using P&D style question-generation and -answering')
parser.add_argument('news_gen_file', help='Path to file used for partition into Question set and Answer set')
parser.add_argument('entity_types', help='Path to freebase entity types')
parser.add_argument('--uu-graphs', help='Path to folder of Unary->Unary entailment graphs to assist question answering')
parser.add_argument('--bu-graphs', help='Path to folder of Binary->Unary entailment graphs to assist question answering')
# parser.add_argument('--typed', action='store_true', help='Boolean flag toggling question typing')
parser.add_argument('--test-all', action='store_true', help='Test all variations of the given configuration')
parser.add_argument('--raw-EGs', action='store_true', help='Read in plain-text entailment graphs from a folder')
parser.add_argument('--local', action='store_true', help='Read in local entailment graphs (default is global)')
parser.add_argument('--eval-fun', default='acc', help='Evaluate results using the specified test function')
# parser.add_argument('--eval-param', type=int, help='Test all variations of the given configuration')
parser.add_argument('--day-range', type=int, default=0, help='Evidence window around the day questions are drawn from (extends # days in both directions)')
parser.add_argument('--plot', action='store_true', help='Plot results visually')

if __name__ == '__main__':
	main()

# Inefficient! Loops over forward entailments to find matches with the question

# def infer_answers(question, query_type, prop_db, ent_graphs, graph_typespace):
# 	answer = set()

# 	if not any(query_type in t for t in ent_graphs.keys()):
# 		return answer

# 	typed_props = [x for x in prop_db if query_type in x.basic_types]

# 	if graph_typespace == EGSpace.ONE_TYPE:
# 		if query_type in ent_graphs:
# 			for prop in typed_props:
# 				ordered_ents = ent_graphs[query_type].get_entailments(prop.pred_desc())
# 				ent_set = {ent.pred for ent in ordered_ents}
# 				if question in ent_set:
# 					answer.add(prop.ARGS[0])

# 	elif graph_typespace == EGSpace.TWO_TYPE:
# 		for prop in typed_props:
# 			graph_type = prop.type_desc()

# 			if graph_type not in ent_graphs:
# 				continue

# 			type_symmetric = prop.basic_types[0] == prop.basic_types[1]

# 			ordered_ents = ent_graphs[graph_type].get_entailments(prop.pred_desc())
# 			rev = False
# 			if not ordered_ents and type_symmetric:
# 				ordered_ents = ent_graphs[graph_type].get_entailments(prop.pred_desc(reverse=True))
# 				rev = True

# 			for x in ordered_ents:
# 				if x.basic_pred == question:
# 					if rev:
# 						pdb.set_trace()
# 					arg_idx = prop.types.index(x.pred.split('#')[1])

# 					answer.add(prop.ARGS[arg_idx])

# 	return answer
