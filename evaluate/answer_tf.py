from collections import defaultdict, Counter, namedtuple
from operator import itemgetter, attrgetter

from proposition import *
from entailment import *
from article import *
from analyze import *
from run import *

from typing import *

def answer_tf_sets(claims_list: List[List[Prop]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
				   uu_graphs: Optional[EGraphCache]=None, bu_graphs: Optional[EGraphCache]=None,
				   A_list: Optional[List[List[bool]]]=None) -> List[List[float]]:
	predictions = []
	for i, cs in enumerate(claims_list):
		answers_list = A_list[i] if A_list else None
		pred_list = answer_tf(cs, evidence_list[i], uu_graphs, bu_graphs, answers=answers_list)
		predictions.append(pred_list)
	return predictions


# Input | questions : [str]
# Input | evidence : [Prop]
# Input | uu_graph : str
# Input | bu_graph : str
# Returns answer sets for each question : [[str]]
def answer_tf(claims: List[Prop], evidence: Tuple[List[Prop], List[Prop]],
			  uu_graphs: Optional[EGraphCache] = None,
			  bu_graphs: Optional[EGraphCache] = None,
			  answers: Optional[List[Set[str]]]=None) -> List[float]:
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
	predictions = []
	prediction_support = []
	for i,c in enumerate(claims):
		q = c.pred_desc()
		prediction_support.append({})
		score = 0

		# Get basic factual answers from observed evidence
		literal_support = [f for f in facts_un[q] if f.args[0] == c.args[0]]
		if literal_support:
			score = 1
			prediction_support[i]['literal'] = literal_support

		# Get inferred answers from U->U graph
		if uu_graphs:
			uu_score, uu_support = infer_claim_UU(c, facts_un, uu_graphs)
			# if uu_score or uu_support:
			# 	assert uu_score and uu_support
			if len(uu_support) > 0:
				score = max(score, uu_score)
				prediction_support[i]['unary'] = uu_support

		# Get inferred answers from B->U graph
		if bu_graphs:
			bu_score, bu_support = infer_claim_BU(c, facts_bin, bu_graphs)
			# if bu_score or bu_support:
			# 	assert bu_score and bu_support
			if len(bu_support) > 0:
				score = max(score, bu_score)
				prediction_support[i]['binary'] = bu_support

		predictions.append(score)

	return predictions


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

	return score, support