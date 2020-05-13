from collections import defaultdict, Counter, namedtuple
from operator import itemgetter, attrgetter

from proposition import *
from entailment import *
from article import *
from analyze import *
from run import *

from typing import *


def answer_question_sets(questions_list: List[List[str]], evidence_list: List[Tuple[List[Prop], List[Prop]]],
						 uu_graphs: Optional[EGraphCache]=None, bu_graphs: Optional[EGraphCache]=None,
						 A_list: Optional[List[List[Set[str]]]]=None) -> \
		Tuple[List[List[List[str]]], List[Dict[str, Any]]]:
	predictions = []
	retro_collation = []
	for i, qs in enumerate(questions_list):
		answers = A_list[i] if A_list else None
		pred_list, retro = answer_questions(qs, evidence_list[i], uu_graphs, bu_graphs, answers=answers)
		predictions.append(pred_list)
		retro_collation.extend(retro)
	return predictions, retro_collation


# Input | questions : [str]
# Input | evidence : [Prop]
# Input | uu_graph : str
# Input | bu_graph : str
# Returns answer sets for each question : [[str]]
def answer_questions(questions: List[str], evidence: Tuple[List[Prop], List[Prop]],
					 uu_graphs: Optional[EGraphCache] = None,
					 bu_graphs: Optional[EGraphCache] = None,
					 answers: Optional[List[Set[str]]]=None) -> Tuple[List[List[str]], List[Dict]]:
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
	prediction_support = []
	for i,q in enumerate(questions):
		answer = Counter()
		prediction_support.append([])
		query_type = q[q.find('#') + 1:]

		# Get basic factual answers from observed evidence
		answer.update(p.args[0] for p in facts_un[q])
		prediction_support[i].append(facts_un[q])

		# Get inferred answers from U->U graph
		if uu_graphs:
			u_ans, u_support = infer_answers(q, query_type, facts_un, uu_graphs, EGSpace.ONE_TYPE)
			answer.update(u_ans)
			prediction_support[i].append(u_support)

		# Get inferred answers from B->U graph
		if bu_graphs:
			b_ans, b_support = infer_answers(q, query_type, facts_bin, bu_graphs, EGSpace.TWO_TYPE)
			answer.update(b_ans)
			prediction_support[i].append(b_support)

		predicted_answers.append(answer)


	ranked_predictions = []
	for c in predicted_answers:
		l = [k for k,v in c.most_common()]
		# l = c.most_common()
		ranked_predictions.append(l)

	# FOR RETROSPECTIVE ANALYSIS ONLY
	retro_collation = []
	if answers:
		possible_support = []
		for i,q in enumerate(questions):
			t = q[q.find('#')+1:]
			a = answers[i]
			us = [p for p in ev_un_ents if t in p.types and p.args[0] in a]
			bs = [p for p in ev_bi_ents if (p.types[0] == t and p.args[0] in a) or (p.types[1] == t and p.args[1] in a)]
			possible_support.append((us, bs))

			ans_choices = {e[:e.find('#')] for p in ev_un_ents + ev_bi_ents for e in p.arg_desc() if e[e.find('#')+1:] == t}
			d = {'question': q,
				 'true_answer': answers[i],
				 'answerable': bool(len(answers[i])),
				 'possible_support': possible_support[i],
				 'prediction': ranked_predictions[i],
				 'prediction_support': prediction_support[i],
				 'possible_choices': len(ans_choices),
				 'exact_q_match': any(q == p.pred_desc() for p in prediction_support[i][0]),
				 }
			d['correct_inference_only'] = any(p in answers[i] for p in ranked_predictions[i]) and not d['exact_q_match']
			d['CORRECT'] = any(p in answers[i] for p in ranked_predictions[i])
			retro_collation.append(d)

		YES = [QA for QA in retro_collation if not QA['answerable'] and QA['prediction'] and not QA['exact_q_match']]
		# if YES:
		# 	print(YES)

	return ranked_predictions, retro_collation

def infer_answers(question: str, query_type: str, prop_cache: Dict[str, List[Prop]],
				  ent_graphs: EGraphCache, graph_typespace: EGSpace) -> Tuple[Counter[str], List[Prop]]:
	answer = Counter()
	support = []

	if not any(query_type in t for t in ent_graphs.keys()):
		return answer, support

	if graph_typespace == EGSpace.ONE_TYPE:
		if query_type in ent_graphs:
			antecedents = ent_graphs[query_type].get_antecedents(question)
			for ant in antecedents:
				# answer |= {p.args[0] for p in prop_cache[ant.pred]}
				ant_support = prop_cache[ant.pred]
				answer.update(p.args[0] for p in ant_support)
				support.extend(ant_support)

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
						support.append(prop)

	return answer, support