from collections import defaultdict, Counter, namedtuple
from operator import itemgetter, attrgetter

from proposition import *
from entailment import *
from article import *
from analyze import *
from run import *
import reference

import pdb
import numpy as np

from typing import *

def answer_tf_sets(claims_list: List[List[Prop]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
				   uu_graphs: Optional[EGraphCache]=None,
				   bu_graphs: Optional[EGraphCache]=None,
				   sim_cache: Optional[EmbeddingCache]=None,
				   A_list: Optional[List[List[bool]]]=None) -> Tuple[List[List[float]], List[List[Dict[str, List[BackEntailment]]]]]:
	predictions = []
	supports = []
	for i, cs in enumerate(claims_list):
		answers_list = A_list[i] if A_list else None
		pred_list, pred_support = answer_tf(cs, evidence_list[i], uu_graphs, bu_graphs, sim_cache=sim_cache, answers=answers_list)
		predictions.append(pred_list)
		supports.append(pred_support)
	if sim_cache and not uu_graphs and not bu_graphs:
		global P_TOTAL, P_NF, P_NAN, H_TOTAL, H_NF, H_NAN
		computed_comparisons = P_TOTAL - P_NF - P_NAN
		computed_questions = H_TOTAL - H_NF - H_NAN
		print('Computed similarities for {}/{} comparisons ({:.1f}%)'.format(computed_comparisons, P_TOTAL, computed_comparisons/P_TOTAL * 100))
		print('Computed answers to {}/{} questions ({:.1f}%)'.format(computed_questions, H_TOTAL, computed_questions / H_TOTAL * 100))
	return predictions, supports


# Input | questions : [str]
# Input | evidence : [Prop]
# Input | uu_graph : str
# Input | bu_graph : str
# Returns answer sets for each question : [[str]]
def answer_tf(claims: List[Prop], evidence: Tuple[List[Prop], List[Prop]],
			  uu_graphs: Optional[EGraphCache] = None,
			  bu_graphs: Optional[EGraphCache] = None,
			  sim_cache: Optional[EmbeddingCache]=None,
			  answers: Optional[List[Set[str]]]=None) -> Tuple[List[float], List[Dict[str, List[BackEntailment]]]]:
	# Keep props containing a named entity
	ev_un_ents = [ev for ev in evidence[0] if 'E' in ev.entity_types]
	ev_bi_ents = [ev for ev in evidence[1] if 'E' in ev.entity_types]

	def _make_prop_cache(props: List[Prop], removals: List[Prop]=[]) -> Dict[str, List[Prop]]:
		cache = defaultdict(list)
		for prop in props:
			cache[prop.pred_desc()].append(prop)
		for r in removals:
			if r.pred_desc() in cache and r in cache[r.pred_desc()]:
				cache[r.pred_desc()].remove(r)
		return cache

	def _make_arg_cache(props: List[Prop], removals: List[Prop]=[]) -> Dict[str, List[Tuple[Prop, int]]]:
		cache = defaultdict(list)
		for prop in props:
			for i, arg in enumerate(prop.args):
				cache[arg].append((prop, i))
		for r in removals:
			for i, arg in enumerate(r.args):
				if arg in cache and (r, i) in cache[arg]:
					cache[arg].remove((r, i))
		return cache

	# Create a prop-indexed fact cache of A: {pred_desc : [prop]} for exact-match and EG lookup
	prop_facts_un = _make_prop_cache(ev_un_ents, removals=claims)
	prop_facts_bi = _make_prop_cache(ev_bi_ents)

	# Create an arg-indexed fact cache of A: {arg : [(prop, arg_idx)]} for similarity lookup
	arg_facts_un = _make_arg_cache(ev_un_ents, removals=claims)
	arg_facts_bi = _make_arg_cache(ev_bi_ents)

	# Return answers to the posed questions
	predictions = []
	prediction_support = []
	for i,c in enumerate(claims):
		if reference.RUNNING_LOCAL and c.types[0] != 'person':
			continue

		q = c.pred_desc()
		prediction_support.append({})
		score = 0

		# Get basic factual answers from observed evidence
		if not sim_cache:
			literal_support = [f for f in prop_facts_un[q] if f.args[0] == c.args[0]]
			if len(literal_support) > 0:
				score = 1
				prediction_support[i]['literal'] = literal_support

		# Get inferred answers from U->U graph
		if uu_graphs:
			uu_score, uu_support = infer_claim_UU(c, prop_facts_un, uu_graphs)
			if len(uu_support) > 0:
				score = max(score, uu_score)
				prediction_support[i]['unary'] = uu_support

		# Get inferred answers from B->U graph
		if bu_graphs:
			bu_score, bu_support = infer_claim_BU(c, prop_facts_bi, bu_graphs)
			if len(bu_support) > 0:
				score = max(score, bu_score)
				prediction_support[i]['binary'] = bu_support

		# Get inferred answers from similarity scores
		if sim_cache and not uu_graphs and not bu_graphs:
			sim_score, sim_support = infer_claim_sim(c, arg_facts_un, arg_facts_bi, sim_cache)
			if sim_support:
				score = max(score, sim_score)
				prediction_support[i]['similarity'] = sim_support

		predictions.append(score)

	return predictions, prediction_support


def infer_claim_UU(claim: Prop, prop_cache: Dict[str, List[Prop]], ent_graphs: EGraphCache) -> \
		Tuple[float, List[BackEntailment]]:

	score = 0
	support: List[BackEntailment] = []

	question = claim.pred_desc()
	query_type = claim.types[0]

	if not any(query_type == t for t in ent_graphs.keys()):
		return score, support

	if query_type in ent_graphs:
		antecedents = ent_graphs[query_type].get_antecedents(question)
		for ant in antecedents:
			ant_support = next((p for p in prop_cache[ant.pred] if p.arg_desc()[0] == claim.arg_desc()[0]), None)
			if ant_support:
				score = max(score, ant.score)
				support.append(ant)

	support.sort(key=lambda ant: ant.score, reverse=True)

	score = 0 if score < 0 else (1 if score > 1 else score)
	return score, support

def infer_claim_BU(claim: Prop, prop_cache: Dict[str, List[Prop]], ent_graphs: EGraphCache) -> \
		Tuple[float, List[BackEntailment]]:

	score = 0
	support: List[BackEntailment] = []

	question = claim.pred_desc()
	query_type = claim.types[0]

	if not any(query_type in t for t in ent_graphs.keys()):
		return score, support

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
					if prop.arg_desc()[arg_idx] == claim.arg_desc()[0]:
						score = max(score, ant.score)
						support.append(ant)
						break

	support.sort(key=lambda ant: ant.score, reverse=True)

	score = 0 if score < 0 else (1 if score > 1 else score)
	return score, support

P_TOTAL = 0
P_NF = 0
P_NAN = 0
H_TOTAL = 0
H_NF = 0
H_NAN = 0
def infer_claim_sim(claim: Prop,
					un_arg_cache: Dict[str, List[Tuple[Prop, int]]],
					bi_arg_cache: Dict[str, List[Tuple[Prop, int]]],
					sim_cache: EmbeddingCache) -> Tuple[float, Optional[str]]:

	score = 0
	support = None
	global P_NF, P_TOTAL, H_NF, H_TOTAL, H_NAN, P_NAN

	arg = claim.args[0]
	if reference.RUNNING_LOCAL and claim.types[0] != 'person':
		return score, support

	H_TOTAL += 1
	hypothesis = claim.prop_desc() + '::0'
	if hypothesis not in sim_cache.id_map:
		H_NF += 1
		return score, support

	h_vec = sim_cache.cache[sim_cache.id_map[hypothesis]]
	if any(np.isnan(h_vec)):
		H_NAN += 1
		return score, support

	h_vecn = h_vec / np.linalg.norm(h_vec)

	for p, arg_idx in un_arg_cache[arg] + bi_arg_cache[arg]:
		if reference.RUNNING_LOCAL and 'person' not in p.types:
			continue

		# Skip binaries with reverse-typing (this is an artifact due to the graphs)
		if len(p.types) == 2 and p.basic_types[0] == p.basic_types[1]:
			if len(p.types[0].split('_')) < 2:
				print('Mismatch of types and basic types: {} / {}'.format(p, p.basic_types))
				continue
			if int(p.types[0].split('_')[-1]) == 2:
				continue

		premise = p.prop_desc() + '::' + str(arg_idx)

		P_TOTAL += 1
		if premise not in sim_cache.id_map:
			P_NF += 1
			continue

		p_vec = sim_cache.cache[sim_cache.id_map[premise]]
		if any(np.isnan(p_vec)):
			P_NAN += 1
			continue

		p_vecn = p_vec / np.linalg.norm(p_vec)
		new_score = np.dot(h_vecn, p_vecn)

		# if premise == hypothesis:
		# 	print('prem/hyp: {} == {} ; sim = {:.3f}'.format(premise, hypothesis, new_score))

		try:
			assert -0.05 <= new_score <= 1.05
		except:
			print('! failed assertion: embedding similarity score out of bounds: {:.5f} (skipping this)'.format(new_score))
			# exit(1)
			continue
		if new_score > score:
			score = new_score
			support = premise
	# pdb.set_trace()
	score = 0 if score < 0 else (1 if score > 1 else score)
	return score, support