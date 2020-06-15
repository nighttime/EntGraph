from collections import defaultdict, Counter, namedtuple
from operator import itemgetter, attrgetter
import datetime

from proposition import *
from entailment import *
from article import *
from analyze import *
from proposition import Prop
from run import TOP_K_ENTS
import reference

from typing import *

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
def generate_questions(Q: List[Prop], aux_Q: List[Prop],
					   pred_cache: Optional[Set[str]],
					   uu_graphs: Optional[EGraphCache],
					   cap: Optional[int]=None) -> \
		Tuple[List[str], List[Set[str]], List[Prop]]:
	# Filter unary props down to just ones containing named entities
	Q_ents = [p for p in Q if 'E' in p.entity_types]
	if len(Q_ents) == 0:
		return [], [], []
	Q_unique_ents = set(e.args[0] for e in Q_ents)
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
	top_props = {}
	for prop in Q_ents:
		if prop.arg_desc()[0] in most_common_ents \
				and prop.pred_desc() in pred_cache:
				# and (count_mentions(prop.args[0], Q_ents)>1 or count_mentions(prop.args[0], aux_evidence)>4):
			top_preds.add(prop.pred_desc())
			top_props[prop.pred_desc()] = prop

	if len(top_preds) == 0:
		return [], [], []

	#	Take all entities that match the common entity preds
	statements = defaultdict(set)
	for prop in Q_ents:
		if prop.pred_desc() in top_preds:  # and prop.types[0] == 'person':
			statements[prop.pred_desc()].add(prop.args[0])

	# Sample the statements and return (question, answers) pairs separated into two lists
	# s = list(statements.items())
	s = [(q, a_set, top_props[q]) for q, a_set in statements.items()]
	random.shuffle(s)
	selected_questions = s[:cap]
	if uu_graphs:
		# filter out questions not answerable using the graphs, if available
		selected_questions = [(q, a, p) for q, a, p in selected_questions if q.split('#')[1] in uu_graphs]
	if True:
		# filer out modals
		selected_questions = [(q, a, p) for q, a, p in selected_questions if not any(q.startswith(m + '.') for m in reference.AUXILIARY_VERBS)]
	questions, answers, props = [list(t) for t in tuple(zip(*selected_questions))]

	answer_choices = {q:{e for p in Q_ents for e in p.arg_desc() if e[e.index('#')+1:] == q[q.index('#')+1:]} for q in questions}

	return questions, answers, props


def generate_question_partitions(articles: List[Article]) -> Tuple[List[List[Article]], Set[str]]:
	pred_counter = Counter()
	for art in articles:
		pred_counter.update([p.pred_desc() for p in art.unary_props])
	top_preds = [(k, v) for k, v in pred_counter.most_common() if k.split('.')[0] not in reference.AUXILIARY_VERBS + reference.LIGHT_VERBS][:5000]
	top_pred_cache = set([pred for pred, count in top_preds])

	article_dates = sorted(set(a.date for a in articles))
	articles_by_day = {d:[a for a in articles if a.date == d and len(a.named_entity_mentions())] for d in article_dates}

	# PARTITION SCHEME: Within N-day span, Articles clustered by entity overlap, Qs drawn from one and answered by the rest
	# global ARGS
	# day_range = ARGS.day_range
	# day_spans = [articles_by_day[i-day_range:i+day_range+1] for i in range(day_range, len(articles_by_day) - day_range)]
	# articles_by_day_span = [sum(ds, []) for ds in day_spans]
	# q_days = articles_by_day[day_range:len(articles_by_day) - day_range]
	#
	# assert len(q_days) == len(articles_by_day_span)
	# for day, span in zip(q_days, articles_by_day_span):
	# 	day_id2art = {a.art_ID: a for a in day}
	#
	# 	# day_feats = {a.art_ID: a.feature_set() for a in day}
	# 	# span_feats = {a.art_ID: (a.feature_set(), a) for a in span}
	#
	# 	# evidence_per_cluster = [[a for j, (feats_j, a) in span_feats.items() if jaccard(feats_i, feats_j)>=0.15 and i != j] for i, feats_i in day_feats.items()]
	#
	# 	day_feats = {a.art_ID: next((f for f in a.feature_counts().most_common() if '#location' not in f[0]), ('____',-1)) for a in day}
	# 	span_feats = {a.art_ID: (a.feature_counts(), a) for a in span}
	#
	# 	clusterings = [
	# 			(feat_i, [day_id2art[i]], [a for j, (feats_j, a) in span_feats.items() if feats_j[feat_i] > 6 and i != j])
	# 			for i, (feat_i, count_i) in day_feats.items() if count_i > 6]
	#
	# 	final_clusterings = [(feat, q, ev) for feat, q, ev in clusterings if ev]
	# 	partitions = [(q, ev) for feat, q, ev in final_clusterings]


	# PARTITION SCHEME: Within 1-day span, Articles clustered by entity overlap, Qs drawn from one and answered by the rest
	# for day in articles_by_day:
	# 	feats = [set(a.feature_counts().keys()) for a in day]
	# 	evidence_per_cluster = [[day[j] for j, feats_j in enumerate(feats) if jaccard(feats_i, feats_j)>0.15 and i != j] for i, feats_i in enumerate(feats)]
	# 	cluster_Q = [[day[i]] for i,c in enumerate(evidence_per_cluster) if len(c)]
	# 	cluster_evidence = [ev for ev in evidence_per_cluster if ev]
	# 	for q, ev in zip(cluster_Q, cluster_evidence):
	# 		partitions.append((q, ev))

	# PARTITION SCHEME: Qs drawn from half-day, E from 5-day surrounding window
	# for days in zip(articles_by_day, articles_by_day[1:], articles_by_day[2:], articles_by_day[3:], articles_by_day[4:]):
	# 	random.shuffle(days[2])
	# 	sep = int(len(days[2])/2)
	# 	# partitions.append((day2[:sep], day1 + day2[sep:] + day3))
	# 	partitions.append((days[2][:sep], days[0] + days[1] + days[2][sep:] + days[3] + days[4]))

	# PARTITION SCHEME: Qs drawn from within article, E from other articles from the same day
	# partitions = [([q_art], [a_art for a_art in day if q_art.art_ID != a_art.art_ID]) for day in articles_by_day[:1] for q_art in day]

	# PARTITION SCHEME: Articles separated by week
	# PARTITION SCHEME: bin articles by ISO week number
	# articles_by_week = defaultdict(list)
	# for a in articles:
	# 	iso_week_number = a.date.isocalendar()[1]
	# 	articles_by_week[iso_week_number].append(a)
	#
	# partitions = []
	# for week in articles_by_week.values():
	# 	random.shuffle(week)
	# 	partitions.append(week)

	day_span = 3

	# article_spans = [articles_by_day[i:i+day_span] for i in range(0, len(articles_by_day) - (day_span - 1), day_span)]
	# partitions = []
	# for span in article_spans:
	# 	p = sum(span, [])
	# 	random.shuffle(p)
	# 	partitions.append(p)

	day_partitions = [[]]
	last_date = article_dates[0]
	for day in article_dates:
		# If day is too far from the current span timeframe, start a new one
		max_date = last_date + datetime.timedelta(days=(day_span-1))
		if max_date < day:
			day_partitions.append([])
			last_date = day
		day_partitions[-1].append(day)

	partitions = [sum([articles_by_day[d] for d in span], []) for span in day_partitions]

	return partitions, top_pred_cache


def generate_positive_question_sets(partitions: List[List[Article]],
									pred_cache: Set[str],
									uu_graphs: Optional[EGraphCache],
									cap: int) -> \
		Tuple[List[List[str]], List[List[Set[str]]], List[List[Prop]], List[Tuple[List[Prop], List[Prop]]]]:

	Q_list, A_list, P_list, evidence_list = [], [], [], []
	for partition in partitions:
		q_unaries = sum((a.unary_props for a in partition), [])
		q_binaries = sum((a.binary_props for a in partition), [])

		q, a, props = generate_questions(q_unaries, q_binaries, pred_cache=pred_cache, uu_graphs=uu_graphs, cap=(cap*len(partition)))
		if not q:
			assert not a and not props
		# for j, q_j in enumerate(q):
		# 	art.remove_qa_pair(q_j, list(a[j])[0])
		# if len(art.unary_props) == 0 and len(art.binary_props) == 0:
		# 	continue
		Q_list.append(q)
		A_list.append(a)
		P_list.append(props)
		evidence_list.append((q_unaries, q_binaries))

	return Q_list, A_list, P_list, evidence_list


# def generate_negative_question_sets(partitions: List[List[Article]],
# 									pred_cache: Set[str],
# 									negative_swaps: Dict[str, Dict[str, Any]],
# 									uu_graphs: EGraphCache,
# 									cap: int) -> List[List[Prop]]:
# 	N_list = []
# 	num_antonyms = 0
#
# 	for partition in partitions:
# 		q_unaries = sum((a.unary_props for a in partition), [])
# 		q_binaries = sum((a.binary_props for a in partition), [])
#
# 		# Pick the K most frequent entities in the question set
# 		Q_props_NE = [p for p in q_unaries if p.entity_types == 'E']
# 		if len(Q_props_NE) == 0:
# 			N_list.append([])
# 			continue
#
# 		ents = Counter(prop.arg_desc()[0] for prop in Q_props_NE)
# 		most_common_ent_counts = ents.most_common(TOP_K_ENTS)
# 		most_common_ents: Set[str] = set(tuple(zip(*most_common_ent_counts))[0])
#
# 		# Cache the local predicates seen with each local entity
# 		ent_pred_mentions = defaultdict(set)
# 		for prop in q_unaries:
# 			ent_pred_mentions[prop.arg_desc()[0]].add(prop.pred_desc())
#
# 		# Create a cache of all mentions of the common ents to speed up the next step
# 		# common_ent_occurrances = defaultdict(set)
# 		# for article in articles:
# 		# 	for prop in article.unary_props:
# 		# 		ent = prop.arg_desc()[0]
# 		# 		if ent in most_common_ents:
# 		# 			common_ent_occurrances[ent].add(prop.pred_desc())
#
# 		# Find predicates never seen globally with candidate ents
# 		# nprops = []
# 		# for ent in most_common_ents:
# 		# 	ent_type = ent.split('#')[1]
# 		# 	npred_candidates = list(pred_cache)
# 		# 	random.shuffle(npred_candidates)
# 		# 	ct = 0
# 		# 	for npred in npred_candidates:
# 		# 		npred_type = npred.split('#')[1]
# 		# 		if ent_type == npred_type and npred not in common_ent_occurrances[ent]:
# 		# 			nprops.append(Prop.from_descriptions(npred, [ent]))
# 		# 			ct += 1
# 		# 			if ct > 5:
# 		# 				break
#
# 		# Substitute adversarial predicates as negatives
# 		nprops = []
# 		for ent in most_common_ents:
# 			ct = 0
# 			max_per_ent = 5
# 			ent_preds = ent_pred_mentions[ent]
# 			for pred in ent_preds:
# 				if ct >= max_per_ent:
# 					break
# 				pred_type = pred.split('#')[1]
# 				if pred_type not in uu_graphs or pred not in uu_graphs[pred_type].nodes:
# 					continue
# 				if pred in negative_swaps:
# 					antonyms = negative_swaps[pred]['antonyms']
# 					troponyms = negative_swaps[pred]['troponyms']
# 					# pred_relations = random.sample(antonyms, len(antonyms)) + random.sample(troponyms, len(troponyms))
# 					pred_relations = random.sample(troponyms, len(troponyms))
# 					for i, relation in enumerate(pred_relations):
# 						# get swapped pred
# 						swapped_pred = swap_pred(pred, relation)
# 						# is swap in same graph?
# 						if swapped_pred not in uu_graphs[pred_type].nodes:
# 							continue
# 						# is swap mentioned at all?
# 						if swapped_pred in ent_pred_mentions[ent]:
# 							continue
# 						# if exists and not mentioned, add to list
# 						nprops.append(Prop.from_descriptions(swapped_pred, [ent]))
# 						ct += 1
# 						if i < len(antonyms):
# 							num_antonyms += 1
# 						if ct >= max_per_ent:
# 							break
#
# 		random.shuffle(nprops)
# 		N_list.append(nprops[:(cap*len(partition))])
#
# 	print('Found {} antonyms and {} troponyms...'.format(num_antonyms, sum(len(n) for n in N_list)-num_antonyms), end=' ', flush=True)
# 	return N_list

def generate_negative_question_sets(P_list: List[List[Prop]],
									partitions: List[List[Article]],
									pred_cache: Set[str],
									negative_swaps: Dict[str, Dict[str, Any]],
									uu_graphs: Optional[EGraphCache],
									cap: int) -> List[List[Prop]]:
	N_list = []
	num_antonyms = 0
	rej = defaultdict(Counter)
	swap_pairs = defaultdict(set)


	for i, ps in enumerate(P_list):
		# Cache the local predicates seen with each local entity
		ent_pred_mentions = defaultdict(set)
		for prop in sum((article.unary_props for article in partitions[i]), []):
			ent_pred_mentions[prop.arg_desc()[0]].add(prop.pred_desc())

		nprops = []
		max_per_positive = 5

		for j, positive_prop in enumerate(ps):
			ct = 0

			pred = positive_prop.pred_desc()
			pred_type = pred.split('#')[1]
			ent = positive_prop.arg_desc()[0]

			if uu_graphs and pred_type not in uu_graphs:
				rej['no graph for predicate type'][pred_type] += 1
				continue

			if pred not in uu_graphs[pred_type].nodes:
				rej['pred not in graph'][pred] += 1
				continue

			if pred not in negative_swaps:
				rej['predicate has no swaps'][pred] += 1
				continue

			antonyms = negative_swaps[pred]['antonyms']
			troponyms = negative_swaps[pred]['troponyms']
			# pred_relations = random.sample(antonyms, len(antonyms)) + random.sample(troponyms, len(troponyms))
			pred_relations = random.sample(troponyms, len(troponyms))

			confirmed_swaps = []
			for a, relation in enumerate(pred_relations):
				# get swapped pred
				swapped_pred = Prop.swap_pred(pred, relation)

				# is swap in same graph?
				if uu_graphs and swapped_pred not in uu_graphs[pred_type].nodes:
					rej['swapped pred is not in the graph'][pred + ' - ' + swapped_pred] += 1
					continue

				# Swap pair is testable
				swap_pairs[pred].add(swapped_pred)

				# is swap mentioned at all?
				if swapped_pred in ent_pred_mentions[ent]:
					rej['swapped pred is actually mentioned'][swapped_pred] += 1
					continue

				# if known and not mentioned, add to list
				confirmed_swaps.append(Prop.from_descriptions(swapped_pred, [ent]))
				# if a < len(antonyms):
				# 	num_antonyms += 1
				ct += 1
				if ct >= max_per_positive:
					break

			if confirmed_swaps:
				nprops.extend(confirmed_swaps)

		random.shuffle(nprops)
		N_list.append(nprops[:(cap*len(partitions[i])*max_per_positive)])

	# print('Found {} antonyms and {} troponyms...'.format(num_antonyms, sum(len(n) for n in N_list)-num_antonyms), end=' ', flush=True)
	return N_list


def generate_tf_question_sets(articles: List[Article], negative_swaps: Optional[Dict[str, Dict[str, Any]]], uu_graphs: Optional[EGraphCache]) -> \
		Tuple[List[List[Prop]], Optional[List[List[Prop]]], List[Tuple[List[Prop], List[Prop]]]]:
	max_questions_per_article = 3
	print('Partitioning...', end=' ', flush=True)
	partitions, top_pred_cache = generate_question_partitions(articles)
	print('Generating positives...', end=' ', flush=True)
	_, _, P_list, evidence_list = generate_positive_question_sets(partitions, top_pred_cache, uu_graphs, cap=max_questions_per_article)
	print('Generating negatives...', end=' ', flush=True)
	N_list = None
	if negative_swaps and uu_graphs:
		# N_list = generate_negative_question_sets(partitions, top_pred_cache, negative_swaps, uu_graphs, cap=max_questions_per_article)
		N_list = generate_negative_question_sets(P_list, partitions, top_pred_cache, negative_swaps, uu_graphs, cap=max_questions_per_article)

	return P_list, N_list, evidence_list


def generate_wh_question_sets(articles: List[Article]) -> \
		Tuple[List[List[str]], List[List[Set[str]]], List[Tuple[List[Prop], List[Prop]]]]:
	partitions, top_pred_cache = generate_question_partitions(articles)
	Q_list, A_list, _, evidence_list = generate_positive_question_sets(partitions, top_pred_cache, cap=3)

	return Q_list, A_list, evidence_list


def format_tf_QA_sets(positives: List[List[Prop]], negatives: List[List[Prop]]) -> Tuple[List[List[Prop]], List[List[int]]]:
	assert len(positives) == len(negatives)
	Q_sets, A_sets = [], []
	for i in range(len(positives)):
		ps = positives[i]
		ns = negatives[i]
		Q_sets.append(ps + ns)
		A_sets.append([1 for _ in ps] + [0 for _ in ns])
		assert len(Q_sets[-1]) == len(A_sets[-1])
	assert len(Q_sets) == len(A_sets)
	return Q_sets, A_sets