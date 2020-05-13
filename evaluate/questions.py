from collections import defaultdict, Counter, namedtuple
from operator import itemgetter, attrgetter
from typing import List

from proposition import *
from entailment import *
from article import *
from analyze import *
from proposition import Prop
from run import *

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
def generate_questions(Q: List[Prop], A: List[Prop], aux_A: List[Prop], cap: Optional[int]=None, pred_cache: Optional[Set[str]]=None) -> \
		Tuple[List[str], List[Set[str]], List[Prop]]:
	# Filter unary props down to just ones containing named entities
	Q_ents = [p for p in Q if 'E' in p.entity_types]
	A_ents = [p for p in (A or []) + (aux_A or []) if 'E' in p.entity_types]
	if len(Q_ents) == 0:
		return [], [], []
	Q_unique_ents = set(e.args[0] for e in Q_ents)
	A_unique_ents = set(e for p in A_ents for e in p.args)
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
	s = [(q, {a for a in a_set if count_mentions(a, A + aux_A) > 0}, top_props[q]) for q, a_set in statements.items()]
	random.shuffle(s)
	questions, answers, props = [list(t) for t in tuple(zip(*s[:cap]))]

	A_ent_choices = {e for p in A_ents for e in p.arg_desc() if e[e.index('#')+1:] in {q[q.index('#')+1:] for q in questions}}

	return questions, answers, props


def generate_question_partitions(articles: List[Article]) -> Tuple[List[Tuple[List[Article], List[Article]]], Set[str]]:
	pred_counter = Counter()
	for art in articles:
		pred_counter.update([p.pred_desc() for p in art.unary_props])
	pred_freqs_ordered = dict((k,v) for k,v in pred_counter.most_common(5000) if not k.startswith('say'))
	top_pred_cache = set([pred for pred, count in pred_freqs_ordered.items()])

	# List of tuples: [ (Q-generating data, evidence data) ]
	partitions: List[Tuple[List[Article], List[Article]]] = []

	article_dates = sorted(set(a.date for a in articles))
	articles_by_day = [[a for a in articles if a.date == d and len(a.named_entity_mentions())] for d in article_dates]

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

	# PARTITION SCHEME: Articles separated by week and randomly split 30%Q/70%A; Qs drawn from Q, E from A
	articles_by_week = defaultdict(list)
	for a in articles:
		iso_week_number = a.date.isocalendar()[1]
		articles_by_week[iso_week_number].append(a)

	partitions = []
	for week in articles_by_week.values():
		random.shuffle(week)
		s = int(len(week) * 0.3)
		partitions.append((week[:s], week[s:]))

	# (Won't work without modification) PARTITION SCHEME: Qs drawn from within article, E from same article
	# partitions = [[a] for a in articles]

	return partitions, top_pred_cache


def generate_positive_question_sets(partitions: List[Tuple[List[Article], List[Article]]], pred_cache: Set[str], cap: int) -> \
		Tuple[List[List[str]], List[List[Set[str]]], List[List[Prop]], List[Tuple[List[Prop], List[Prop]]]]:

	Q_list, A_list, P_list, evidence_list = [], [], [], []
	for partition in partitions:
		question_data, evidence_data = partition
		q_unaries = sum((a.unary_props for a in question_data), [])
		q_binaries = sum((a.binary_props for a in question_data), [])

		e_unaries = sum((a.unary_props for a in evidence_data), [])
		e_binaries = sum((a.binary_props for a in evidence_data), [])

		q, a, props = generate_questions(q_unaries, e_unaries, e_binaries, cap=(cap*len(question_data)), pred_cache=pred_cache)
		if not q:
			assert not a and not props
		# for j, q_j in enumerate(q):
		# 	art.remove_qa_pair(q_j, list(a[j])[0])
		# if len(art.unary_props) == 0 and len(art.binary_props) == 0:
		# 	continue
		Q_list.append(q)
		A_list.append(a)
		P_list.append(props)
		evidence_list.append((e_unaries, e_binaries))

	return Q_list, A_list, P_list, evidence_list


def generate_negative_question_sets(partitions: List[Tuple[List[Article], List[Article]]], articles: List[Article], pred_cache: Set[str], cap: int) -> \
		List[List[Prop]]:
	N_list = []

	for partition in partitions:
		question_data, evidence_data = partition
		q_unaries = sum((a.unary_props for a in question_data), [])
		q_binaries = sum((a.binary_props for a in question_data), [])

		e_unaries = sum((a.unary_props for a in evidence_data), [])
		e_binaries = sum((a.binary_props for a in evidence_data), [])

		# Pick the K most frequent entities in the question set
		Q_props_NE = [p for p in q_unaries if p.entity_types == 'E' and count_mentions(p.args[0], e_unaries + e_binaries)]
		if len(Q_props_NE) == 0:
			N_list.append([])
			continue

		ents = Counter(prop.arg_desc()[0] for prop in Q_props_NE)
		most_common_ent_counts = ents.most_common(TOP_K_ENTS)
		most_common_ents = set(tuple(zip(*most_common_ent_counts))[0])

		# Create a cache of all mentions of the common ents to speed up the next step
		common_ent_occurrances = defaultdict(set)
		for article in articles:
			for prop in article.unary_props:
				ent = prop.arg_desc()[0]
				if ent in most_common_ents:
					common_ent_occurrances[ent].add(prop.pred_desc())

		# Find predicates never seen globally with candidate ents
		nprops = []
		for ent in most_common_ents:
			ent_type = ent.split('#')[1]
			npred_candidates = list(pred_cache)
			random.shuffle(npred_candidates)
			ct = 0
			for npred in npred_candidates:
				npred_type = npred.split('#')[1]
				if ent_type == npred_type and npred not in common_ent_occurrances[ent]:
					nprops.append(Prop.from_descriptions(npred, [ent]))
					ct += 1
					if ct > 5:
						break

		random.shuffle(nprops)
		N_list.append(nprops[:(cap*len(question_data))])

	return N_list


def generate_tf_question_sets(articles: List[Article]) -> \
		Tuple[List[List[Prop]], List[List[Prop]], List[Tuple[List[Prop], List[Prop]]]]:
	max_questions_per_article = 3
	print('Partitioning...', end=' ', flush=True)
	partitions, top_pred_cache = generate_question_partitions(articles)
	print('Generating positives...', end=' ', flush=True)
	_, _, P_list, evidence_list = generate_positive_question_sets(partitions, top_pred_cache, cap=max_questions_per_article)
	print('Generating negatives...', end=' ', flush=True)
	N_list = generate_negative_question_sets(partitions, articles, top_pred_cache, cap=max_questions_per_article)

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