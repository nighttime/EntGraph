import argparse
import sys
import os
import pdb
import json
import re
from collections import defaultdict, Counter
from operator import itemgetter
from functools import reduce
import itertools
import random
from datetime import datetime
import numpy as np

from proposition import *
from entailment import *
from article import *
from analyze import *

from typing import *


# For printing
BAR_LEN = 50
BAR = '=' * BAR_LEN
bar = '-' * BAR_LEN

# Program Hyperparameters
# Fraction of props partitioned into Q
QA_SPLIT = 0.5
TOP_K_ENTS = 100


# Input | source_fname : str
# Returns dataset processed and partitioned into Q and A : ({int:Article}, {int:Article})
def partitionQA(unary_props, binary_props):
	u_cutoff = int(len(unary_props) * QA_SPLIT)
	b_cutoff = int(len(binary_props) * QA_SPLIT)
	props_Q = unary_props[:u_cutoff]
	props_A_u = unary_props[u_cutoff:]
	props_A_b = binary_props[b_cutoff:]

	return props_Q, (props_A_u, props_A_b)

def count_mentions(arg: str, props: List[Prop]) -> int:
	return len([p for p in props if arg in p.args])

# Input | Q : [Prop]
# Input | cap : int or None (for no max cap)
# Returns questions (typed preds) and answer sets : ([str], [{str}])
def generate_questions(Q: List[Prop], aux_evidence: List[Prop], cap: Optional[int]=None, pred_cache: Optional[Set[str]]=None) -> \
		Tuple[List[str], List[Set[str]]]:
	# Filter unary props down to just ones containing named entities
	Q_ents = [p for p in Q if 'E' in p.entity_types]
	if len(Q_ents) == 0:
		return [], []
	unique_ents = set(e.args[0] for e in Q_ents)
	# print('  {:.1f}% of Q is named entities ({} instances, {} unique)'.format(len(Q_ents)/len(Q)*100, len(Q_ents), len(unique_ents)))

	# Generate index of: entity + type -> count
	ents = Counter(prop.arg_desc()[0] for prop in Q_ents)

	# Pick the K most frequent entities
	most_common_ents = ents.most_common(TOP_K_ENTS)
	most_common_ents = set(tuple(zip(*most_common_ents))[0])

	# Generate questions from the most common entities: pred -> {prop}
	#	Keep predicates that match an entity
	# top_preds = {prop.pred_desc() for prop in Q_ents if prop.arg_desc()[0] in most_common_ents and prop.pred_desc() in pred_cache and count_mentions(prop.args[0], Q_ents + aux_evidence)>1}
	top_preds = set()
	for prop in Q_ents:
		if prop.arg_desc()[0] in most_common_ents \
				and prop.pred_desc() in pred_cache:
				# and (count_mentions(prop.args[0], Q_ents)>1 or count_mentions(prop.args[0], aux_evidence)>4):
			top_preds.add(prop.pred_desc())

	if len(top_preds) == 0:
		return [], []

	#	Take all entities that match the common entity preds
	statements = defaultdict(set)
	for prop in Q_ents:
		if prop.pred_desc() in top_preds:  # and prop.types[0] == 'person':
			statements[prop.pred_desc()].add(prop.args[0])

	# Sample the statements and return (question, answers) pairs separated into two lists
	s = list(statements.items())
	random.shuffle(s)
	questions, answers = tuple(zip(*s[:cap]))
	return questions, answers


def generate_question_sets(articles: List[Article], cap: Optional[int] = None) -> \
		Tuple[List[List[str]], List[List[Set[str]]], List[Tuple[List[Prop], List[Prop]]]]:
	pred_counter = Counter()
	for art in articles:
		pred_counter.update([p.pred_desc() for p in art.unary_props])
	stop = 30
	top_pred_cache = set([pred for pred, count in pred_counter.most_common(5000 + stop)][stop:])

	partitions: List[Tuple[List[Article], List[Article]]] = []

	article_dates = sorted(set(a.date for a in articles))
	articles_by_day = [[a for a in articles if a.date == d and len(a.named_entity_mentions())] for d in article_dates]

	for days in zip(articles_by_day, articles_by_day[1:], articles_by_day[2:], articles_by_day[3:], articles_by_day[4:]):
		random.shuffle(days[2])
		sep = int(len(days[2])/2)
		# partitions.append((day2[:sep], day1 + day2[sep:] + day3))
		partitions.append((days[2][:sep], days[0] + days[1] + days[2][sep:] + days[3] + days[4]))

	# partitions = [[a] for a in articles]

	Q_list, A_list, evidence_list = [], [], []
	for partition in partitions:
		question_data, evidence_data = partition
		q_unaries = sum((a.unary_props for a in question_data), [])
		q_binaries = sum((a.binary_props for a in question_data), [])

		q, a = generate_questions(q_unaries, q_binaries, cap=cap, pred_cache=top_pred_cache)
		if not q:
			continue
		# for j, q_j in enumerate(q):
		# 	art.remove_qa_pair(q_j, list(a[j])[0])
		# if len(art.unary_props) == 0 and len(art.binary_props) == 0:
		# 	continue
		Q_list.append(q)
		A_list.append(a)

		e_unaries = sum((a.unary_props for a in evidence_data), [])
		e_binaries = sum((a.binary_props for a in evidence_data), [])

		evidence_list.append((e_unaries, e_binaries))
	return Q_list, A_list, evidence_list


# Input | questions : [str]
# Input | evidence : [Prop]
# Input | uu_graph : str
# Input | bu_graph : str
# Returns answer sets for each question : [[str]]
def answer_questions(questions: List[str], evidence: Tuple[List[Prop], List[Prop]],
					 uu_graphs: Optional[EGraphCache] = None,
					 bu_graphs: Optional[EGraphCache] = None,
					 answers: Optional[List[Set[str]]]=None) -> List[List[str]]:
	# Keep props containing a named entity
	# ev_un_ents, ev_bi_ents = tuple(list(filter(lambda x: 'E' in x.entity_types, evidence)) for i in range(2))
	ev_un_ents = [ev for ev in evidence[0] if 'E' in ev.entity_types]
	ev_bi_ents = [ev for ev in evidence[1] if 'E' in ev.entity_types]

	def _make_prop_cache(props: List[Prop]) -> Dict[str, List[Prop]]:
		cache = defaultdict(list)
		for prop in props:
			cache[prop.pred_desc()].append(prop)
		return cache

	# Create a cached fact index of A: {pred_desc : [prop]}
	facts_un = _make_prop_cache(ev_un_ents)
	facts_bin = _make_prop_cache(ev_bi_ents)

	# Return answers to the posed questions
	predicted_answers = []
	for q in questions:
		answer = Counter()
		query_type = q[q.find('#') + 1:]

		# Get basic factual answers from observed evidence
		answer.update(p.args[0] for p in facts_un[q])

		# Get inferred answers from U->U graph
		if uu_graphs:
			u_ans = infer_answers(q, query_type, facts_un, uu_graphs, EGSpace.ONE_TYPE)
			answer.update(u_ans)

		# Get inferred answers from B->U graph
		if bu_graphs:
			b_ans = infer_answers(q, query_type, facts_bin, bu_graphs, EGSpace.TWO_TYPE)
			answer.update(b_ans)

		predicted_answers.append(answer)

	ranked_predictions = []
	for c in predicted_answers:
		l = [k for k,v in c.most_common()]
		# l = c.most_common()
		ranked_predictions.append(l)

	return ranked_predictions


def answer_question_sets(questions_list: List[List[str]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
						 uu_graphs: Optional[EGraphCache]=None, bu_graphs: Optional[EGraphCache]=None,
						 A_list: Optional[List[List[Set[str]]]]=None) -> List[List[List[str]]]:
	predictions = []
	for i, qs in enumerate(questions_list):
		pred_list = answer_questions(qs, evidence_list[i], uu_graphs, bu_graphs)
		predictions.append(pred_list)
	return predictions


def infer_answers(question: str, query_type: str, prop_cache: Dict[str, List[Prop]],
				  ent_graphs: EGraphCache, graph_typespace: EGSpace) -> Counter[str]:
	answer = Counter()

	if not any(query_type in t for t in ent_graphs.keys()):
		return answer

	if graph_typespace == EGSpace.ONE_TYPE:
		if query_type in ent_graphs:
			antecedents = ent_graphs[query_type].get_antecedents(question)
			for ant in antecedents:
				# answer |= {p.args[0] for p in prop_cache[ant.pred]}
				answer.update(p.args[0] for p in prop_cache[ant.pred])

	elif graph_typespace == EGSpace.TWO_TYPE:
		graph_types = {p.type_desc() for props in prop_cache.values() for p in props}

		for graph_type in graph_types:
			if graph_type not in ent_graphs:
				continue

			type_symmetric = len({*graph_type.split('#')}) == 1
			suffixes = ['_1', '_2'] if type_symmetric else ['']

			for suffix in suffixes:
				qualified_question = question + suffix
				qualified_question_type = qualified_question.split('#')[1]
				antecedents = ent_graphs[graph_type].get_antecedents(qualified_question)
				for ant in antecedents:
					for prop in prop_cache[ant.pred]:
						arg_idx = prop.types.index(qualified_question_type)
						# answer.add(prop.args[arg_idx])
						answer[prop.args[arg_idx]] += 1

	return answer


def eval_sets(A_list: List[List[Set[str]]], predictions_list: List[List[List[str]]],
			  eval_fun: Callable[[List[Set[str]], List[List[str]]], float]) -> float:
	A_list_flat = []
	P_list_flat = []
	for i, art_list in enumerate(A_list):
		for j, ans_list in enumerate(art_list):
			A_list_flat.append(ans_list)
			P_list_flat.append(predictions_list[i][j])

	assert len(A_list_flat) == len(P_list_flat)
	return eval_fun(A_list_flat, P_list_flat)


def run(questions, answers, evidence, score_text, uu_graphs=None, bu_graphs=None, eval_fun=ans):
	print(bar)
	predictions = answer_questions(questions, evidence, uu_graphs=uu_graphs, bu_graphs=bu_graphs)
	score = eval_fun(answers, predictions)
	print(score_text.format(score))
	checkpoint()


def run_sets(Q_list: List[List[str]], A_list: List[List[Set[str]]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
			 score_text: str, eval_fun: Callable[[List[Set[str]], List[List[str]]], float],
			 uu_graphs: EGraphCache=None, bu_graphs: EGraphCache=None):
	print(bar)
	predictions_list = answer_question_sets(Q_list, evidence_list, uu_graphs=uu_graphs, bu_graphs=bu_graphs)
	score = eval_sets(A_list, predictions_list, eval_fun=eval_fun)
	print(score_text)
	print('{:.4f}'.format(score))
	checkpoint()


def checkpoint():
	print('+ Checkpoint:', datetime.now().strftime('%H:%M:%S'))


def main():
	args = parser.parse_args()

	eval_fun = globals()[args.eval_fun]

	print()
	print(BAR)
	print('Running Eval')

	original_types_file = 'data/freebase_types/entity2Types.txt'
	print('* Using entity types from: ' + args.entity_types)

	resources = ['literal answers']
	if args.uu_graphs is not None:
		resources.append('U->U graphs')
	if args.bu_graphs is not None:
		resources.append('B->U graphs')
	print('* Using evidence from: ' + str(resources))

	print('* Evaluating with: ' + eval_fun.__name__)

	checkpoint()
	print(BAR)

	#######################################

	print('Loading entity type cache...')
	if not os.path.exists(args.entity_types):
		print('  !generating new cache...')
		load_entity_types(original_types_file)
		save_precomputed_entity_types(args.entity_types)
	load_precomputed_entity_types(args.entity_types)

	# Reading in news articles
	articles, unary_props, binary_props = read_source_data(args.news_gen_file)

	# Generate questions from Q
	print('Generating questions...', end=' ', flush=True)
	Q_list, A_list, evidence_list = generate_question_sets(articles, cap=100)
	print('Generated {} questions from {} sets'.format(sum(len(qs) for qs in Q_list), len(Q_list)))

	# arts = articles
	# dates = set(a.date for a in arts)
	# days = sorted(dates)[:7]
	# arts_days = [a for a in arts if a.date in days and len(a.named_entity_mentions())]
	# arts_days_fsets = [set(a.named_entity_mentions()) for a in arts_days]
	# clusters = []
	# jaccard = lambda x,y: len(x.intersection(y))/len(x.union(y))
	# for featset in arts_days_fsets:
	# 	similar = [i for i,f in enumerate(arts_days_fsets) if jaccard(featset, f)>=0.3]
	# 	clusters.append(similar)
	# lens = [len(x) for x in clusters]



	uu_graphs, bu_graphs = None, None

	# Open U->U entailment graphs
	if args.uu_graphs:
		print('Reading U->U graphs...', end=' ', flush=True)
		if args.raw_EGs:
			uu_graphs = read_graphs(args.uu_graphs)
		else:
			uu_graphs = load_precomputed_EGs(args.uu_graphs)
		print('Graphs read: {}'.format(len(uu_graphs)))

	# Open B->U entailment graphs
	if args.bu_graphs:
		print('Reading B->U graphs...', end=' ', flush=True)
		if args.raw_EGs:
			bu_graphs = read_graphs(args.bu_graphs)
		else:
			bu_graphs = load_precomputed_EGs(args.bu_graphs)
		print('Graphs read: {}'.format(len(bu_graphs)))

	checkpoint()
	print(BAR)

	# Answer the questions using available resources: A set, U->U Graph, B->U graph
	print('Predicting answers...')

	if args.test_all:
		run_sets(Q_list, A_list, evidence_list, 'baseline score:', eval_fun=eval_fun)

		if uu_graphs:
			run_sets(Q_list, A_list, evidence_list, 'With U->U score:', eval_fun=eval_fun, uu_graphs=uu_graphs)

		if bu_graphs:
			run_sets(Q_list, A_list, evidence_list, 'With B->U score:', eval_fun=eval_fun, bu_graphs=bu_graphs)

		if uu_graphs and bu_graphs:
			run_sets(Q_list, A_list, evidence_list, 'With U->U and B->U score:', eval_fun=eval_fun,
					 uu_graphs=uu_graphs, bu_graphs=bu_graphs)

	else:
		run_sets(Q_list, A_list, evidence_list, 'score:', eval_fun=eval_fun, uu_graphs=uu_graphs, bu_graphs=bu_graphs)

	print(BAR)
	print()


# ts = list(map(lambda x: x.split('#')[1], questions))
# hist = {t:ts.count(t)/len(ts) for t in set(ts)}

# run_sets(Q_list, A_list, evidence_list, 'With U->U and B->U score: {:.5f}', uu_graphs=uu_graphs, bu_graphs=bu_graphs, eval_fun=eval_fun)
# print(bar)
# counters_uu_bu = [Counter(dict(preds_uu[i])) + Counter(dict(preds_bu[i])) for i in range(len(preds_uu))]
# preds_uu_bu = [c.most_common() for c in counters_uu_bu]
# score = evaluate(true_answers, preds_uu_bu)
# print('With U->U and B->U score: {:.5f}'.format(score))
# checkpoint()


parser = argparse.ArgumentParser(description='Evaluate using P&D style question-generation and -answering')
parser.add_argument('news_gen_file', help='Path to file used for partition into Question set and Answer set')
parser.add_argument('entity_types', help='Path to freebase entity types')
parser.add_argument('--uu-graphs', help='Path to folder of Unary->Unary entailment graphs to assist question answering')
parser.add_argument('--bu-graphs', help='Path to folder of Binary->Unary entailment graphs to assist question answering')
parser.add_argument('--typed', action='store_true', help='Boolean flag toggling question typing')
parser.add_argument('--test-all', action='store_true', help='Test all variations of the given configuration')
parser.add_argument('--raw-EGs', action='store_true', help='Read in plain-text entailment graphs from a folder')
parser.add_argument('--eval-fun', default='MRR', help='Evaluate results using the specified test function')
parser.add_argument('--eval-param', type=int, help='Test all variations of the given configuration')

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
# 					answer.add(prop.args[0])

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

# 					answer.add(prop.args[arg_idx])

# 	return answer
