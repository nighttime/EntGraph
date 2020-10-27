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
				   models: Dict[str, Any],
				   answer_modes: Set[str],
				   A_list: Optional[List[List[bool]]]=None) -> Tuple[List[List[float]], List[List[Dict[str, Prop]]], List[Prop]]:
	predictions = []
	supports = []
	for i, cs in enumerate(claims_list):
		answers_list = A_list[i] if A_list else None
		pred_list, pred_support = answer_tf(cs, evidence_list[i], models, answer_modes, answers=answers_list)
		predictions.append(pred_list)
		supports.append(pred_support)
	if 'Sim' in answer_modes and not any(m in answer_modes for m in ['UU', 'BU', 'BB']):
		global P_TOTAL, P_NF, P_NAN, P_FETCH_ERR, H_TOTAL, H_NF, H_NAN, H_FETCH_ERR
		computed_comparisons = P_TOTAL - P_NF - P_NAN
		computed_questions = H_TOTAL - H_NF - H_NAN
		# print('Computed similarities for {}/{} comparisons ({:.1f}%)'.format(computed_comparisons, P_TOTAL, computed_comparisons/P_TOTAL * 100))
		# print('Computed answers to {}/{} questions ({:.1f}%)'.format(computed_questions, H_TOTAL, computed_questions / H_TOTAL * 100))
		print('H_TOTAL {}\t H_NF {}\t H_NAN {}\t H_FETCH_ERR {}'.format(H_TOTAL, H_NF, H_NAN, H_FETCH_ERR))
		print('P_TOTAL {}\t P_NF {}\t P_NAN {}\t P_FETCH_ERR {}'.format(P_TOTAL, P_NF, P_NAN, P_FETCH_ERR))

	global ERROR_CASES

	return predictions, supports, ERROR_CASES


# Input | questions : [str]
# Input | evidence : [Prop]
# Input | uu_graph : str
# Input | bu_graph : str
# Returns answer sets for each question : [[str]]
def answer_tf(claims: List[Prop], evidence: Tuple[List[Prop], List[Prop]],
			  models: Dict[str, Any],
			  answer_modes: Set[str],
			  answers: Optional[List[Set[str]]]=None) -> Tuple[List[float], List[Dict[str, Prop]]]:
	# Keep props containing a named entity
	ev_un = [ev for ev in evidence[0] if 'E' in ev.entity_types]
	ev_bi = [ev for ev in evidence[1] if 'E' in ev.entity_types]

	def _make_prop_cache(props: List[Prop], removals: List[Prop]=[]) -> Dict[str, List[Prop]]:
		cache = defaultdict(list)
		for prop in props:
			cache[prop.pred_desc()].append(prop)
		for r in removals:
			if r.pred_desc() in cache and r in cache[r.pred_desc()]:
				cache[r.pred_desc()].remove(r)
		return cache

	def _make_u_arg_cache(props: List[Prop], removals: List[Prop]=[]) -> Dict[str, List[Tuple[Prop, int]]]:
		cache = defaultdict(list)
		for prop in props:
			for i, arg in enumerate(prop.args):
				cache[arg].append((prop, i))
		for r in removals:
			for i, arg in enumerate(r.args):
				if arg in cache and (r, i) in cache[arg]:
					cache[arg].remove((r, i))
		return cache

	def _make_b_arg_cache(props: List[Prop], removals: List[Prop]=[]) -> Dict[str, List[Prop]]:
		cache = defaultdict(list)
		for prop in props:
			args = tuple(sorted(prop.args))
			cache[args].append(prop)
		for r in removals:
			args = tuple(sorted(r.args))
			if args in cache and r in cache[args]:
				cache[args].remove(r)
		return cache

	# Create a prop-indexed fact cache of A: {pred_desc : [prop]} for exact-match and EG lookup
	prop_facts_u = _make_prop_cache(ev_un, removals=claims)
	prop_facts_b = _make_prop_cache(ev_bi, removals=claims)

	# Create an arg-indexed fact cache of A: {arg : [(prop, arg_idx)]} for similarity lookup
	arg_facts_u = _make_u_arg_cache(ev_un + ev_bi, removals=claims)
	arg_facts_b = _make_b_arg_cache(ev_bi, removals=claims)

	# Return answers to the posed questions
	predictions = []
	prediction_support = []
	for c in claims:
		if reference.RUNNING_LOCAL and 'person' not in c.basic_types:
			continue

		q = c.pred_desc()
		is_unary = len(c.args) == 1
		prediction_support.append({})
		score = 0

		# Append all possible answer supports for later analysis
		if is_unary:
			available_support  = [f for f in arg_facts_u[c.args[0]] if f[0].args[f[1]] == c.args[0]]
		else:
			sorted_cargs = tuple(sorted(c.args))
			available_support  = [f for f in arg_facts_b[sorted_cargs] if tuple(sorted(f.args)) == sorted_cargs]
		prediction_support[-1]['Available'] = available_support

		# Get basic factual answers from observed evidence
		if 'Literal U' in answer_modes:
			literal_support = [f for f in prop_facts_u[q] if f.args == c.args]
			if len(literal_support) > 0:
				score = 1
				prediction_support[-1]['Literal U'] = literal_support

		if 'Literal B' in answer_modes:
			literal_support = [f for f in prop_facts_b[q] if f.args == c.args]
			if len(literal_support) > 0:
				score = 1
				prediction_support[-1]['Literal B'] = literal_support

		# Get inferred answers from U->U graph
		if 'UU' in answer_modes and is_unary:
			uu_score, uu_support = infer_claim_UU(c, prop_facts_u, models['UU'])
			if uu_support:
				score = max(score, uu_score)
				prediction_support[-1]['UU'] = uu_support

		# Get inferred answers from B->B/U graph
		if 'BU' in answer_modes and is_unary:
			bu_score, bu_support = infer_claim_BU(c, prop_facts_b, models['BU'])
			if bu_support:
				score = max(score, bu_score)
				prediction_support[-1]['BU'] = bu_support

		# Get inferred answers from B->B/U graph
		if 'BB' in answer_modes and not is_unary:
			bb_score, bb_support = infer_claim_BB(c, prop_facts_b, models['BU'])
			if bb_support:
				score = max(score, bb_score)
				prediction_support[-1]['BB'] = bb_support

		# Get inferred answers from similarity scores
		if 'BERT' in answer_modes:
			sim_score, sim_support = infer_claim_sim(c, arg_facts_u, arg_facts_b, models['BERT'])
			if sim_support:
				score = max(score, sim_score)
				prediction_support[-1]['BERT'] = sim_support
		if 'RoBERTa' in answer_modes:
			sim_score, sim_support = infer_claim_sim(c, arg_facts_u, arg_facts_b, models['RoBERTa'])
			if sim_support:
				score = max(score, sim_score)
				prediction_support[-1]['RoBERTa'] = sim_support

		predictions.append(score)

	return predictions, prediction_support


def infer_claim_UU(claim: Prop, prop_cache: Dict[str, List[Prop]], ent_graphs: EGraphCache) -> \
		Tuple[float, Optional[Prop]]:

	score = 0
	support = None

	question = claim.pred_desc()
	query_type = claim.types[0]

	if not any(query_type == t for t in ent_graphs.keys()):
		return score, support

	if query_type in ent_graphs:
		antecedents = ent_graphs[query_type].get_antecedents(question)
		for ant in antecedents:
			ant_support = next((p for p in prop_cache[ant.pred] if p.arg_desc()[0] == claim.arg_desc()[0]), None)
			if ant_support and ant.score > score:
				score = ant.score
				support = ant_support

	score = 0 if score < 0 else (1 if score > 1 else score)
	return score, support

def infer_claim_BU(claim: Prop, prop_cache: Dict[str, List[Prop]], ent_graphs: EGraphCache) -> \
		Tuple[float, Optional[Prop]]:

	score = 0
	support = None

	question = claim.pred_desc()
	query_type = claim.types[0]

	if not any(query_type in t for t in ent_graphs.keys()):
		return score, support

	graph_types = {p.type_desc() for props in prop_cache.values() for p in props if query_type in p.basic_types}

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
						if ant.score > score:
							score = max(score, ant.score)
							support = prop
							break

	score = 0 if score < 0 else (1 if score > 1 else score)
	return score, support

def infer_claim_BB(claim: Prop, prop_cache: Dict[str, List[Prop]], ent_graphs: EGraphCache) -> \
		Tuple[float, Optional[Prop]]:

	score = 0
	support = None

	question = claim.pred_desc()
	query_type = claim.type_desc()

	if query_type not in ent_graphs:
		return score, support

	antecedents = ent_graphs[query_type].get_antecedents(question)
	for ant in antecedents:
		ant_support = next((p for p in prop_cache[ant.pred] if p.arg_desc() == claim.arg_desc() or list(reversed(p.arg_desc())) == claim.arg_desc()), None)
		if ant_support and ant.score > score:
			score = ant.score
			support = ant_support

	score = 0 if score < 0 else (1 if score > 1 else score)
	return score, support

P_TOTAL = 0
P_NF = 0
P_NAN = 0
P_FETCH_ERR = 0
H_TOTAL = 0
H_NF = 0
H_NAN = 0
H_FETCH_ERR = 0
sim_err_count = 0
ERROR_CASES = []
def infer_claim_sim(claim: Prop,
					u_arg_cache: Dict[str, List[Tuple[Prop, int]]],
					b_arg_cache: Dict[str, List[Prop]],
					sim_cache: EmbeddingCache) -> Tuple[float, Optional[Prop]]:

	score = 0
	support = None
	global P_TOTAL, P_NF, P_NAN, P_FETCH_ERR, H_TOTAL, H_NF, H_NAN, H_FETCH_ERR
	global sim_err_count
	global ERROR_CASES

	if reference.RUNNING_LOCAL and claim.types[0] != 'person':
		return score, support

	H_TOTAL += 1
	hypothesis = claim.prop_desc() # + '::0'
	if hypothesis not in sim_cache.id_map:
		H_NF += 1
		H_FETCH_ERR += 1
		if H_NF < 10:
			print('H NF:', hypothesis)
		return score, support

	h_vec = sim_cache.cache[sim_cache.id_map[hypothesis]]
	if any(np.isnan(h_vec)):
		H_NAN += 1
		H_FETCH_ERR += 1
		ERROR_CASES.append(claim)
		if H_NAN < 50:
			print('H NaN:', hypothesis)
		return score, support

	h_vecn = h_vec / np.linalg.norm(h_vec)

	is_unary = len(claim.args) == 1
	if is_unary:
		args = claim.args[0]
		premises = [p for p, i in u_arg_cache[args]]
	else:
		args = tuple(sorted(claim.args))
		premises = b_arg_cache[args]

	prem_nf_ct = 0
	prem_nan_ct = 0
	for p in premises:
		if reference.RUNNING_LOCAL and 'person' not in p.types:
			continue

		# Skip binaries with reverse-typing (this is an artifact due to the graphs)
		if len(p.types) == 2 and p.basic_types[0] == p.basic_types[1]:
			if len(p.types[0].split('_')) < 2:
				print('Mismatch of types and basic types: {} / {}'.format(p, p.basic_types))
				continue
			if int(p.types[0].split('_')[-1]) == 2:
				continue

		premise = p.prop_desc() # + '::' + str(arg_idx)

		if premise == hypothesis:
			return 1.0, p

		P_TOTAL += 1
		if premise not in sim_cache.id_map:
			P_NF += 1
			prem_nf_ct += 1
			continue

		p_vec = sim_cache.cache[sim_cache.id_map[premise]]
		if any(np.isnan(p_vec)):
			P_NAN += 1
			prem_nan_ct += 1
			continue

		p_vecn = p_vec / np.linalg.norm(p_vec)
		cos_score = np.dot(h_vecn, p_vecn)

		new_score = (1 + cos_score) / 2

		# if premise == hypothesis:
		# 	print('prem/hyp: {} == {} ; sim = {:.3f}'.format(premise, hypothesis, new_score))

		try:
			assert -0.05 <= new_score <= 1.05
		except:
			print('! failed assertion: embedding similarity score out of bounds: {:.5f} (clamping this value)'.format(new_score))
			# exit(1)
			# continue
		if new_score > score:
			score = new_score
			support = p

	if len(premises) == prem_nan_ct + prem_nf_ct:
		P_FETCH_ERR += 1

	# if score == 0 and sim_err_count < 30:
	# 	ps = [p.prop_desc() for p in premises]
	# 	print('! No answer found for {} amongst {}'.format(hypothesis, ps))
	# 	sim_err_count += 1

	score = 0 if score < 0 else (1 if score > 1 else score)
	return score, support