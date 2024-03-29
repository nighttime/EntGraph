import math
from collections import defaultdict, Counter, namedtuple
from operator import itemgetter, attrgetter

import proposition
import utils
from graph_encoder import GraphDeducer
from proposition import *
from entailment import *
from article import *
from analyze import *
from questions import prop_in_graphs
from run_mv import *
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
		utils.print_progress(i / len(claims_list), 'completed')
		if not cs:
			predictions.append([])
			supports.append([])
			continue
		answers_list = A_list[i] if A_list else None
		pred_list, pred_support = answer_tf(i, cs, evidence_list[i], models, answer_modes, answers=answers_list)
		predictions.append(pred_list)
		supports.append(pred_support)
	utils.print_progress(1.0, 'completed')

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

# Input | questions : [str]
# Input | evidence : [Prop]
# Input | uu_graph : str
# Input | bu_graph : str
# Returns answer sets for each question : [[str]]
def answer_tf(set_num: int, claims: List[Prop], evidence: Tuple[List[Prop], List[Prop]],
			  models: Dict[str, Any],
			  answer_modes: Set[str],
			  answers: Optional[List[Set[str]]]=None) -> Tuple[List[float], List[Dict[str, Prop]]]:
	# Keep props containing a named entity
	ev_un = [ev for ev in evidence[0] if 'E' in ev.entity_types]
	ev_bi = [ev for ev in evidence[1] if 'E' in ev.entity_types]

	# Create a prop-indexed fact cache of A: {pred_desc : [prop]} for exact-match and EG lookup
	prop_facts_u = _make_prop_cache(ev_un, removals=claims)
	prop_facts_b = _make_prop_cache(ev_bi, removals=claims)

	# Create an arg-indexed fact cache of A: {arg : [(prop, arg_idx)]} for similarity lookup
	arg_facts_u = _make_u_arg_cache(ev_un + ev_bi, removals=claims)
	arg_facts_b = _make_b_arg_cache(ev_bi, removals=claims)

	# Return answers to the posed questions
	predictions = []
	prediction_support = []
	for j, c in enumerate(claims):
		if reference.RUNNING_LOCAL and 'UU' in models and 'BU' in models and 'person' not in c.basic_types:
			continue

		q = c.pred_desc()
		is_unary = len(c.args) == 1
		prediction_support.append({})
		score = 0.0

		# Append all possible answer supports for later analysis
		if is_unary:
			# available_support  = [f for f in arg_facts_u[c.args[0]] if f[0].args[f[1]] == c.args[0]]
			available_support = [f for f,i in arg_facts_u[c.args[0]] if f.args[i] == c.args[0]]
		else:
			sorted_cargs = tuple(sorted(c.args))
			available_support  = [f for f in arg_facts_b[sorted_cargs] if tuple(sorted(f.args)) == sorted_cargs]
		prediction_support[-1]['Available'] = available_support

		available_support_u = [p for p in available_support if len(p.args) == 1]
		available_support_b = [p for p in available_support if len(p.args) == 2]

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

		if 'Lemma Baseline' in answer_modes:
			if models['Lemma Baseline'][set_num] == 1.0:
				score = 1
				prediction_support[-1]['Lemma Baseline'] = evidence[0]

		# q_found = prop_in_graphs(c, models['UU'], models['BU'])

		# Get inferred answers from U->U graph
		if 'UU' in answer_modes and is_unary:
			uu_score, uu_support = infer_claim_UU(c, prop_facts_u, arg_facts_u, models['UU'])
			if uu_support and 'UU-LM' not in answer_modes:
				score = max(score, uu_score)
				prediction_support[-1]['UU'] = uu_support
			elif 'UU-LM' in answer_modes:
				uulm_score, uulm_support, _ = deduce_edge(c, available_support_u, models, 'UU-LM')
				if uulm_support:
					score = max(score, uulm_score)
					prediction_support[-1]['UU-LM'] = uulm_support

		# Get inferred answers from B->B/U graph
		if 'BU' in answer_modes and is_unary:
			bu_score, bu_support = infer_claim_BU(c, prop_facts_b, arg_facts_u, models['BU'])
			if bu_support and 'BU-LM' not in answer_modes:
				score = max(score, bu_score)
				prediction_support[-1]['BU'] = bu_support
			elif 'BU-LM' in answer_modes:
				bulm_score, bulm_support, _ = deduce_edge(c, available_support_b, models, 'BU-LM')
				if bulm_support:
					score = max(score, bulm_score)
					prediction_support[-1]['BU-LM'] = bulm_support

		# Get inferred answers from B->B/U graph
		if 'BB' in answer_modes and not is_unary:
			bb_score, bb_support = infer_claim_BB(c, prop_facts_b, arg_facts_b, models['BU'])

			cond1_always_use_EG_support = bb_support and reference.SMOOTHING_TRIGGER != reference.TRIGGER.NO_EG_UNCOMBINED
			cond1_use_EG_support_UNCOMB = bb_support and 'BB-LM' not in answer_modes and reference.SMOOTHING_TRIGGER == reference.TRIGGER.NO_EG_UNCOMBINED

			cond2_no_EG = not bb_support and 'BB-LM' in answer_modes and reference.SMOOTHING_TRIGGER == reference.TRIGGER.NO_EG_SUPPORT
			cond2_no_EG_UNCOMB = not bb_support and 'BB-LM' in answer_modes and reference.SMOOTHING_TRIGGER == reference.TRIGGER.NO_EG_UNCOMBINED
			cond2_always = 'BB-LM' in answer_modes and reference.SMOOTHING_TRIGGER == reference.TRIGGER.ALWAYS

			# if bb_support:
			# if bb_support and 'BB-LM' not in answer_modes:
			if any([cond1_always_use_EG_support, cond1_use_EG_support_UNCOMB]):
				score = max(score, bb_score)
				prediction_support[-1]['BB'] = bb_support

				# elif not bb_support and 'BB-LM' in answer_modes:
				# elif 'BB-LM' in answer_modes:
				# if 'BB-LM' in answer_modes:
			if any([cond2_no_EG, cond2_no_EG_UNCOMB, cond2_always]):
				bblm_score, bblm_support, expansions = deduce_edge(c, available_support_b, models, 'BB-LM')
				prediction_support[-1]['BB-LM-Expansions'] = expansions
				if bblm_support:
					score = max(score, bblm_score)
					prediction_support[-1]['BB-LM'] = bblm_support

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

		if 'PPDB' in answer_modes:
			sim_score, sim_support = infer_claim_ppdb(c, arg_facts_u, arg_facts_b, models['PPDB'])
			if sim_support:
				score = max(score, sim_score)
				prediction_support[-1]['PPDB'] = sim_support

		predictions.append(score)

	# assert len(predictions) == len(claims)
	return predictions, prediction_support


def infer_claim_UU(claim: Prop, prop_cache: Dict[str, List[Prop]], arg_cache: Dict[str, List[Tuple[Prop, int]]], ent_graphs: EGraphCache) -> \
		Tuple[float, Optional[Prop]]:

	score = 0
	support = None

	question = claim.pred_desc()
	query_type = claim.types[0]

	# if not any(query_type == t for t in ent_graphs.keys()):
	# 	return score, support

	if query_type not in ent_graphs:
		return score, support

	antecedents = ent_graphs[query_type].get_antecedents(question)
	for ant in antecedents:
		ant_support = next((p for p in prop_cache[ant.pred] if p.arg_desc()[0] == claim.arg_desc()[0]), None)
		if ant_support and ant.score > score:
			score = ant.score
			support = ant_support

	# Back off to other graphs
	backoff_node = reference.GRAPH_BACKOFF == 'node' and len(antecedents) == 0
	backoff_edge = reference.GRAPH_BACKOFF == 'edge' and support is None

	if backoff_node or backoff_edge:
		antecedents_list = query_all_graphs_for_prop(claim, ent_graphs)
		evidence_props = arg_cache[claim.args[0]]
		found_edges = defaultdict(int)
		score_sum = defaultdict(float)
		for antecedents in antecedents_list:
			for ant in antecedents:
				for p,i in evidence_props:
					if len(p.args) == 1 and ant.pred.split('#')[0] == p.pred:
						found_edges[p] += 1
						score_sum[p] += ant.score

		backoff_scores = {k: score_sum[k] / found_edges[k] for k, v in found_edges.items()}
		for p, s in backoff_scores.items():
			if s > score:
				score = s
				support = p

	score = 0 if score < 0 else (1 if score > 1 else score)
	return score, support

def infer_claim_BU(claim: Prop, prop_cache: Dict[str, List[Prop]], arg_cache: Dict[str, List[Tuple[Prop, int]]], ent_graphs: EGraphCache) -> \
		Tuple[float, Optional[Prop]]:

	score = 0
	support = None

	question = claim.pred_desc()
	query_type = claim.types[0]

	if not any(query_type in t for t in ent_graphs.keys()):
		return score, support

	graph_types = {p.type_desc() for props in prop_cache.values() for p in props if query_type in p.basic_types}

	found_node = False

	for graph_type in graph_types:
		if graph_type not in ent_graphs:
			continue

		type_symmetric = len({*graph_type.split('#')}) == 1
		suffixes = ['_1', '_2'] if type_symmetric else ['']

		for suffix in suffixes:
			qualified_question = question + suffix
			qualified_question_type = qualified_question.split('#')[1]
			antecedents = ent_graphs[graph_type].get_antecedents(qualified_question)
			if len(antecedents) > 0:
				found_node = True
			for ant in antecedents:
				for prop in prop_cache[ant.pred]:
					arg_idx = prop.types.index(qualified_question_type)
					l_arg = prop.arg_desc()[arg_idx]
					r_arg = claim.arg_desc()[0]
					if l_arg == r_arg:
						if ant.score > score:
							score = max(score, ant.score)
							support = prop
							break

	# Back off to other graphs
	backoff_node = reference.GRAPH_BACKOFF == 'node' and not found_node
	backoff_edge = reference.GRAPH_BACKOFF == 'edge' and support is None

	if backoff_node or backoff_edge:
		antecedents_list = query_all_graphs_for_prop(claim, ent_graphs)
		evidence_props = arg_cache[claim.args[0]]
		found_edges = defaultdict(int)
		score_sum = defaultdict(float)
		for antecedents in antecedents_list:
			for ant in antecedents:
				for p, i in evidence_props:
					if len(p.args) == 2 and ant.pred.split('#')[0] == p.pred:
						found_edges[p] += 1
						score_sum[p] += ant.score

		backoff_scores = {k: score_sum[k] / found_edges[k] for k, v in found_edges.items()}
		for p, s in backoff_scores.items():
			if s > score:
				score = s
				support = p

	score = 0 if score < 0 else (1 if score > 1 else score)
	return score, support

def infer_claim_BB(claim: Prop, prop_cache: Dict[str, List[Prop]], arg_cache: Dict[str, List[Prop]], ent_graphs: EGraphCache) -> \
		Tuple[float, Optional[Prop]]:
	score = 0
	support = None

	question = claim.pred_desc()
	query_type = claim.type_desc()

	##############################
	### Changed for L/H dataset processing (should be the same for MV eval but just in case...)
	# if query_type not in ent_graphs:
	# 	return score, support
	#
	# antecedents = ent_graphs[query_type].get_antecedents(question)
	#
	# for ant in antecedents:
	# 	ant_support = next((p for p in prop_cache[ant.pred] if p.arg_desc() == claim.arg_desc() or list(reversed(p.arg_desc())) == claim.arg_desc()), None)
	# 	if ant_support and ant.score > score:
	# 		score = ant.score
	# 		support = ant_support
	##############################
	antecedents = set()
	if query_type in ent_graphs:
		antecedents = ent_graphs[query_type].get_antecedents(question)
		for ant in antecedents:
			ant_support = next((p for p in prop_cache[ant.pred] if p.arg_desc() == claim.arg_desc() or list(reversed(p.arg_desc())) == claim.arg_desc()), None)
			if ant_support and ant.score > score:
				score = ant.score
				support = ant_support
	##############################

	# Back off to other graphs
	node_not_found = len(antecedents) == 0 # detect if hypothesis is in the graph
	##############################
	# edge_not_found = support is None
	##############################
	edge_not_found = support is None or node_not_found
	##############################
	# need to detect if the premise is in the graph
	evidence_props = arg_cache[tuple(sorted(claim.args))]
	no_premises_found = not any(p.pred_desc() in ent_graphs['#'.join(p.types)].nodes for p in evidence_props if '#'.join(p.types) in ent_graphs)
	# no_premises_found = not any(ent_graphs['#'.join(p.types)].get_entailments(p.pred_desc()) for p in evidence_props if '#'.join(p.types) in ent_graphs)
	both_nodes_not_found = node_not_found or no_premises_found

	backoff_node = node_not_found and reference.GRAPH_BACKOFF == 'node'
	backoff_edge = edge_not_found and reference.GRAPH_BACKOFF == 'edge'
	backoff_both_nodes = both_nodes_not_found and reference.GRAPH_BACKOFF == 'both_nodes'

	if backoff_node or backoff_edge or backoff_both_nodes:
		antecedents_list = query_all_graphs_for_prop(claim, ent_graphs)
		evidence_props = arg_cache[tuple(sorted(claim.args))]
		found_edges = defaultdict(int)
		score_sum = defaultdict(float)
		for antecedents in antecedents_list:
			for ant in antecedents:
				for p in evidence_props:
					if ant.pred.split('#')[0] == p.pred:
						found_edges[p] += 1
						score_sum[p] += ant.score

		backoff_scores = {k:score_sum[k]/found_edges[k] for k,v in found_edges.items()}
		for p,s in backoff_scores.items():
			if s > score:
				score = s
				support = p

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
	hypothesis = claim.prop_desc()
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

		premise = p.prop_desc()

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


def infer_claim_ppdb(claim: Prop,
					u_arg_cache: Dict[str, List[Tuple[Prop, int]]],
					b_arg_cache: Dict[str, List[Prop]],
					ppdb: Dict[str, Dict[str, float]]) -> Tuple[float, Optional[Prop]]:
	score = 0
	support = None

	hypothesis = format_prop_ppdb_lookup(claim)

	is_unary = len(claim.args) == 1
	if is_unary:
		args = claim.args[0]
		premise_props = [p for p, i in u_arg_cache[args]]
	else:
		args = tuple(sorted(claim.args))
		premise_props = b_arg_cache[args]


	found_hyp = False
	best_h_text = ''
	best_p_text = ''
	while len(hypothesis.split()) > 0 and not found_hyp:
		if hypothesis in ppdb:
			found_hyp = True

		for p in premise_props:
			if reference.RUNNING_LOCAL and 'person' not in p.types:
				continue

			# Skip binaries with reverse-typing (this is an artifact due to the graphs)
			if len(p.types) == 2 and p.basic_types[0] == p.basic_types[1]:
				if len(p.types[0].split('_')) < 2:
					print('Mismatch of types and basic types: {} / {}'.format(p, p.basic_types))
					continue
				if int(p.types[0].split('_')[-1]) == 2:
					continue

			if claim.prop_desc() == p.prop_desc():
				# print('ppdb: {} => {} / {} => {}'.format(premise, hypothesis, p, claim))
				return 1.0, p

			premise = format_prop_ppdb_lookup(p)
			found_prem = False
			while len(premise.split()) > 0 and not found_prem:

				if premise in ppdb:
					found_prem = True

				if found_hyp:
					if premise in ppdb[hypothesis]:
						new_score = ppdb[hypothesis][premise]
						if new_score > score:
							score = new_score
							support = p
							best_h_text = hypothesis
							best_p_text = premise

				premise = ' '.join(premise.split()[:-1])

		hypothesis = ' '.join(hypothesis.split()[:-1])


	normed_score = score / 100.0
	# if normed_score > 0:
	# 	print('ppdb: {} => {} / {} => {}'.format(best_p_text, best_h_text, support, claim))
	return normed_score, support

def pred_length(p: str) -> int:
	start = p.index('(') + 1
	stop = p.index(')')
	pred = p[start:stop]
	return len(pred)

# q_start : pred_desc, q_cand : pred_desc
def good_Q_cand(q_start: str, q_cand: str) -> bool:
	if q_cand == q_start:
		return True

	# If cand is negated start
	if q_cand == 'NEG__' + q_start:
		return False

	stop_words = {'be', 'get', 'have', 'do', 'go', 'give', 'take', 'make', 'let'}
	def get_words(pred: str) -> Set[str]:
		return {w for w in '.'.join(pred.split('(')[1].split(')')[0].split(',')).split('.') if w not in {'1', '2', '3'}}

	q_start_words = get_words(q_start)
	q_cand_words = get_words(q_cand)

	# If cand is just a stop-word
	if len(q_cand_words) == 1 and list(q_cand_words)[0] in stop_words and len(q_start_words) > 1:
		return False

	# If cand is a truncated version of start
	if q_cand_words.issubset(q_start_words) and len(q_start_words) > len(q_cand_words):
		return False

	# If cand contains a generic stop-word replacement
	for w in stop_words:
		if w in q_cand_words and w not in q_start_words:
			if (q_cand_words - {w}).issubset(q_start_words):
				return False

	return True


def deduce_edge(q: Prop, evidence: List[Prop], models: Dict[str, Any], answer_mode: str) -> Tuple[float, Optional[Prop], List[Tuple[str, str, float]]]:
	# Parameters
	# k = 'logscale'
	# k = 'proportional'
	k = reference.SMOOTHING_K
	smooth_P = reference.SMOOTH_P
	smooth_Q = reference.SMOOTH_Q

	positional_weighting = False

	score = 0
	support = None
	all_expansions_edges = []
	# analyze the expansions of X in position P and Q, and what edges they lead to. Should lead to many FPs in Q position and mostly TPs in P position

	if not evidence:
		return score, support, all_expansions_edges
	evidence = list(set(evidence))

	deducer_l = models[{'BB-LM': 'BV-B-Deducer', 'BU-LM': 'BV-B-Deducer', 'UU-LM': 'UV-U-Deducer'}[answer_mode]]
	deducer_r = models[{'BB-LM': 'BV-B-Deducer', 'BU-LM': 'BV-U-Deducer', 'UU-LM': 'UV-U-Deducer'}[answer_mode]]

	graph = models['BU'] if answer_mode in ['BB-LM', 'BU-LM'] else models['UU']

	graph_uu = models['UU'] if 'UU' in graph else None
	graph_bu = models['BU'] if 'BU' in graph else None

	nearest_pred_clusters_l, nearest_score_clusters_l = ([], [])

	ps_by_type = defaultdict(list)
	for p in evidence:
		ps_by_type[tuple(sorted(p.basic_types))].append(p)

	ps_grouped = ps_by_type.values()
	ps_ordered = [p for ps in ps_grouped for p in ps]
	for group in ps_grouped:
		if smooth_P == reference.SMOOTH.ALL:
			to_expand_P = [True] * len(group)
		elif smooth_P == reference.SMOOTH.MISSING:
			to_expand_P = [not prop_in_graphs(p, graph_uu, graph_bu) for p in group]
		elif smooth_P == reference.SMOOTH.NONE:
			to_expand_P = [False] * len(group)

		preds_to_expand = [p.pred_desc() for i, p in enumerate(group) if to_expand_P[i]]
		preds_not_to_expand = [p.pred_desc() for i, p in enumerate(group) if not to_expand_P[i]]

		if 'WN-P' in models:
			preds_to_expand = [p for p in preds_to_expand if not p.startswith('(be.1,be.in.2')]
			ps_base = [proposition.extract_predicate_base_term(p) for p in preds_to_expand]
			# ps_replacements = [models['WN-P'][b]['hypernyms'] if b in models['WN-P'] else [] for b in ps_base]
			# ps_repl_filter_syn = [list(set(ps_replacements[i]) - set(models['WN-P'][b]['synonyms'])) if b in models['WN-P'] else [] for i,b in enumerate(ps_base)]
			# new_ps = [[Prop.swap_pred(p, new_p_base) for new_p_base in ps_replacements[i]] if ps_replacements[i] else [preds_to_expand[i]] for i,p in enumerate(preds_to_expand)]
			# new_ps = [[Prop.swap_pred(p, new_p_base) for new_p_base in ps_repl_filter_syn[i]] if ps_repl_filter_syn[i] else [preds_to_expand[i]] for i, p in enumerate(preds_to_expand)]

			ps_replacements = []
			for b in ps_base:
				if b in models['WN-P']:
					# Filter out synonyms if we are not actively using them
					if reference.P_WN_RELATION == reference.WN_RELATION.SYNONYM:
						replacements = [p for p in models['WN-P'][b][reference.P_WN_RELATION.value]]
					else:
						replacements = [p for p in models['WN-P'][b][reference.P_WN_RELATION.value] if p not in models['WN-P'][b]['synonyms']]
					ps_replacements.append(replacements)
				else:
					ps_replacements.append([])

			new_ps = [[Prop.swap_pred(p, new_p_base) for new_p_base in ps_replacements[i]] if ps_replacements[i] else [p] for i,p in enumerate(preds_to_expand)]
			new_scores = [[1.0]*len(ps) for ps in new_ps]
			res_ps = (new_ps, new_scores)
		elif 'hyp-tree-P' in models:
			pred_typings = ['#'.join(p.split('#')[1:]).replace('_1', '').replace('_2', '') for p in preds_to_expand]
			if pred_typings and pred_typings[0] not in models['hyp-tree-P']:
				return score, support, all_expansions_edges
			# graph_trees = [models['hyp-tree-P'][pred_typings[i]][0] for i in range(len(preds_to_expand))]
			# new_ps = [[p for p,score in graph_trees[i][p] if p in graph_trees[i]][:k] for i,p in enumerate(preds_to_expand)]
			if preds_to_expand:
				graph_tree = models['hyp-tree-P'][pred_typings[0]][0]
				new_p_cands = [[(p, 1.0)] for p in preds_to_expand]
				seen_ps = [set() for p in preds_to_expand]
				for i in range(1):
					new_p_cands = [[(p2,s*s2) for p, s in pgroup for p2, s2 in (graph_tree[p] if p in graph_tree else [])] for pgroup in new_p_cands]
					# expand only to unseen predicates
					# new_p_cands = [[(p2, s * s2) for p, s in new_p_cands[i] for p2, s2 in (graph_tree[p] if p in graph_tree else []) if p2 not in seen_ps[i]]
					# 		  for i in range(len(new_p_cands))]
					# new_p_cands = [sorted(pgroup, reverse=True, key=lambda x: x[1])[:k] for pgroup in new_p_cands]
					# seen_ps = [seen_ps[i] | set([p for p,s in new_p_cands[i]]) for i in range(len(seen_ps))]
				new_ps = [[p for p,s in sorted(pgroup, reverse=True, key=lambda x: x[1])[:k]] for pgroup in new_p_cands]
			else:
				new_ps = []
			new_scores = [[1.0]*len(ps) for ps in new_ps]
			res_ps = (new_ps, new_scores)
		else:
			res_ps = deducer_l.get_nearest_node(preds_to_expand, k=k, available_graphs=set(graph.keys()))
			# force lm-found neighbors to have score of 1.0
			if res_ps:
				res_ps = (res_ps[0], [[1]*len(ps) for ps in res_ps[1]])

		if res_ps:
			deducer_l.log['{} LHS: nn found'.format(answer_mode)] += 1
			nearest_pred_clusters_l.extend(res_ps[0])
			nearest_score_clusters_l.extend(res_ps[1])
		else:
			deducer_l.log['{} LHS: nn not found'.format(answer_mode)] += 1
			nearest_pred_clusters_l.extend([[p] for p in preds_to_expand])
			nearest_score_clusters_l.extend([[1.0] for _ in range(len(preds_to_expand))])

		if preds_not_to_expand:
			nearest_pred_clusters_l.extend([[p] for p in preds_not_to_expand])
			nearest_score_clusters_l.extend([[1.0] for _ in range(len(preds_not_to_expand))])

		# res_ps_nn = deducer_l.get_nearest_node(preds_to_expand, k=k, available_graphs=set(graph.keys()))
		# if res_ps_nn:
		# 	deducer_l.log['{} LHS: nn found'.format(answer_mode)] += 1
		# 	nearest_pred_clusters_l.extend(res_ps_nn[0])
		# 	nearest_score_clusters_l.extend(res_ps_nn[1])
		# else:
		# 	deducer_l.log['{} LHS: nn not found'.format(answer_mode)] += 1
		# 	nearest_pred_clusters_l.extend([[p] for p in preds_to_expand])
		# 	nearest_score_clusters_l.extend([[1.0] for _ in range(len(preds_to_expand))])
		#
		# if preds_not_to_expand:
		# 	nearest_pred_clusters_l.extend([[p] for p in preds_not_to_expand])
		# 	nearest_score_clusters_l.extend([[1.0] for _ in range(len(preds_not_to_expand))])

	# res_ps_nn = deducer_l.get_nearest_node([p.pred_desc() for p in evidence], k=k, model_typings=set(graph.keys()))

	# if not res_ps_nn and not any(ps_in_graph):
	# 	return score, support

	# nearest_pred_clusters, nearest_score_clusters = res_ps_nn if res_ps_nn else ([], [])
	# nearest_pred_clusters.extend([[p.pred_desc()] for i,p in enumerate(evidence) if ps_in_graph[i]])
	# nearest_score_clusters.extend([[1.0] for i, p in enumerate(evidence) if ps_in_graph[i]])
	# nearest_pred_clusters = [[p.pred_desc()] for p in evidence]
	# nearest_score_clusters = [[1.0] for _ in range(len(nearest_pred_clusters))]

	res_r = ([q.pred_desc()], [1.0])

	expand_Q = False
	if smooth_Q == reference.SMOOTH.ALL:
		expand_Q = True
	elif smooth_Q == reference.SMOOTH.MISSING:
		expand_Q = not prop_in_graphs(q, graph_uu, graph_bu)
	elif smooth_Q == reference.SMOOTH.NONE:
		expand_Q = False

	if expand_Q and answer_mode != 'BU-LM':
		if 'WN-Q' in models:
			q_base = proposition.extract_predicate_base_term(q.pred)
			# Filter out synonyms if we are not actively using them
			if reference.Q_WN_RELATION == reference.WN_RELATION.SYNONYM:
				q_replacements = [p for p in models['WN-Q'][q_base][reference.Q_WN_RELATION.value]] if q_base in models['WN-Q'] else []
			else:
				q_replacements = [p for p in models['WN-Q'][q_base][reference.Q_WN_RELATION.value] if
								  p not in models['WN-Q'][q_base]['synonyms']] if q_base in models['WN-Q'] else []
				bad_troponyms = ['get', 'give', 'take', 'have', 'see', 'meet', 'make', 'say', 'set', 'move', 'be'] # 'put', 'hold', 'keep'
				q_replacements = [p for p in q_replacements if p not in bad_troponyms]

			new_qs = [Prop.swap_pred(q.pred_desc(), new_q_base) for new_q_base in q_replacements] or res_r[0]
			new_scores = [1.0] * len(new_qs)
			res_r = (new_qs, new_scores)
		elif 'hyp-tree-Q' in models:
			typing = '#'.join(q.basic_types)
			if typing not in models['hyp-tree-Q']:
				return score, support, all_expansions_edges
			graph_tree_hypo = models['hyp-tree-Q'][typing][1]
			graph_tree_hyper = models['hyp-tree-Q'][typing][0]

			if q.pred_desc() in graph_tree_hypo and pred_length(q.pred) < pred_length(ps_ordered[0].pred):
				q_cands = [p for p, score in graph_tree_hypo[q.pred_desc()]]
				# q_cands = [p for p in q_cands if p not in [p_ for p_, score_ in graph_tree_hyper[q.pred_desc()]]]
				# new_qs = [p for p in q_cands if good_Q_cand(q.pred_desc(), p)][:k]
				new_qs = q_cands[:k]
			else:
				new_qs = []

			# Sort and filter qs by length (may not be helpful)
			# new_qs = [p for p, score in graph_tree_hypo[q.pred_desc()]] if q.pred_desc() in graph_tree_hypo else []
			# new_qs = sorted([nq for nq in new_qs if pred_length(nq) > pred_length(q.pred_desc())], reverse=True, key=pred_length)[:k]

			# if q.pred_desc() in graph_tree_hypo:
			# 	new_qs = [p for p, score in graph_tree_hypo[q.pred_desc()] if not (q.pred_desc() != p and p in graph_tree_hypo and q.pred_desc() in [_q for _q, _s in graph_tree_hypo[p]])]
			# 	new_qs = new_qs[:k]
			# else:
			# 	new_qs = []
			new_scores = [1.0] * len(new_qs)
			if new_qs:
				res_r = (new_qs, new_scores)
		else:
			res_q_nn = deducer_r.get_nearest_node([q.pred_desc()], k=k, available_graphs=set(graph.keys()))
			if res_q_nn:
				# res_r = (res_q_nn[0][0], res_q_nn[1][0])
				res_r = (res_q_nn[0][0], [1]*len(res_q_nn[1][0]))
				deducer_r.log['{} RHS: nn found'.format(answer_mode)] += 1
			else:
				deducer_r.log['{} RHS: nn not found'.format(answer_mode)] += 1

	for i in range(len(nearest_pred_clusters_l)):
		preds = nearest_pred_clusters_l[i]
		scores = nearest_score_clusters_l[i]
		p = ps_ordered[i]

		if not preds:
			continue

		res_l = (preds, scores)

		basic_typing = '#'.join(preds[0].split('#')[1:]).replace('_1', '').replace('_2', '')

		if expand_Q and answer_mode == 'BU-LM':
			res_q_nn = deducer_r.get_nearest_node([q.pred_desc()], k=k, available_graphs=set(graph.keys()), target_typing=basic_typing)
			if res_q_nn:
				res_r = (res_q_nn[0][0], res_q_nn[1][0])
				deducer_r.log['{} RHS: nn found'.format(answer_mode)] += 1
			else:
				deducer_r.log['{} RHS: nn not found'.format(answer_mode)] += 1

		for lhs_pred, lhs_score in zip(*res_l):
			lhs = Prop.with_new_pred(p, lhs_pred)
			if positional_weighting:
				try:
					lhs_score *= (1 / (1 + math.log10(1 + graph[basic_typing].get_entailments_count(lhs_pred))))
				except KeyError:
					lhs_score *= 1 / 4

			for rhs_pred, rhs_score in zip(*res_r):
				rhs = Prop.with_new_pred(q, rhs_pred)

				if positional_weighting:
					try:
						rhs_score *= (1 / (1 + math.log10(1 + len(graph[basic_typing].get_antecedents(rhs_pred)))))
					except KeyError:
						rhs_score *= 1 / 4

				possible_score, possible_support = 0, None
				if answer_mode == 'UU-LM':
					prop_facts_u = _make_prop_cache([lhs], removals=[rhs])
					arg_facts_u = _make_u_arg_cache([lhs], removals=[rhs])
					possible_score, possible_support = infer_claim_UU(rhs, prop_facts_u, arg_facts_u, graph)

				elif answer_mode == 'BB-LM':
					prop_facts_b = _make_prop_cache([lhs], removals=[rhs])
					arg_facts_b = _make_b_arg_cache([lhs], removals=[rhs])
					possible_score, possible_support = infer_claim_BB(rhs, prop_facts_b, arg_facts_b, graph)

				elif answer_mode == 'BU-LM':
					prop_facts_b = _make_prop_cache([lhs], removals=[rhs])
					arg_facts_u = _make_u_arg_cache([lhs], removals=[rhs])
					possible_score, possible_support = infer_claim_BU(rhs, prop_facts_b, arg_facts_u, graph)

				all_expansions_edges.append((lhs.pred_desc(), rhs.pred_desc(), possible_score))
				if possible_support:
					possible_score = possible_score * lhs_score * rhs_score
					# possible_score = (bb_score * min(lhs_score, rhs_score))

					if possible_score > score:
						score = possible_score
						support = lhs

	# if answer_mode == 'UU-LM':
	# 	score *= 2

	return score, support, all_expansions_edges