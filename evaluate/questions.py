from collections import defaultdict, Counter, namedtuple
from operator import itemgetter, attrgetter
import datetime
import re

from proposition import *
from entailment import *
from article import *
from analyze import *
import proposition
from proposition import Prop
import reference
import random
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

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
def generate_questions(partition: List[Article],
					   pred_cache: Optional[Dict[str, int]],
					   question_modes: Set[str],
					   uu_graphs: Optional[EGraphCache],
					   bu_graphs: Optional[EGraphCache]) -> Tuple[List[Prop], List[Prop], List[Prop]]:
	un_Q = [p for a in partition for p in a.unary_props]
	bin_Q = [p for a in partition for p in a.selected_binary_props]
	bin_Q_all = [p for a in partition for p in a.binary_props]
	num_arts = len(partition)
	num_sents = sum(len(a.sents) for a in partition)

	# cap = int(0.1 * num_arts) if reference.RUNNING_LOCAL else 1 * num_arts
	cap = 1 * len(partition)

	# Filter out general entity props (keep named entities)
	un_Q_NE = [p for p in un_Q if 'E' in p.entity_types]
	bin_Q_NE = [p for p in bin_Q if 'E' in p.entity_types]


	########## FILTER OUT BAD PROPS

	# ---------- Unary ----------
	un_Q_cand = un_Q_NE

	# Filter out questions not answerable using the graphs, if available (essentially for local debug mode)
	if uu_graphs:
		un_Q_cand = [p for p in un_Q_cand if p.types[0] in uu_graphs]

	# Filter out duplicate props
	un_Q_cand = list(set(un_Q_cand))

	# ---------- Binary ----------
	bin_Q_cand = bin_Q_NE

	# Filter out prepositions
	bin_Q_cand = [p for p in bin_Q_cand if proposition.extract_predicate_base_term(p.pred) not in reference.PREPOSITIONS]

	# Filter out questions not answerable using the graphs, if available (essentially for local debug mode)
	if bu_graphs:
		bin_Q_cand = [p for p in bin_Q_cand if '#'.join(p.basic_types) in bu_graphs]

	# Filter out duplicate props
	bin_Q_cand = list(set(bin_Q_cand))


	# Collect binaries by root predicate to filter out duplicate events
	# WILL NOT WORK... NEED TO CLUSTER LINE BY LINE AT ARTICLE LEVEL
	# bin_Q_roots = defaultdict(list)
	# for p in bin_Q_cand:
	# 	root = proposition.binary_pred_root(p)
	# 	bin_Q_roots[(root, p.args)].append(p)

	########## IDENTIFY CANDIDATE ENTITIES

	# Generate counts of mentions for each unique entity (across binaries and unaries)
	ent_counts_u = Counter(prop.arg_desc()[0] for prop in un_Q_NE)
	ent_counts_bu  = Counter(prop.arg_desc()[0] for prop in bin_Q_NE if prop.entity_types[0] == 'E')
	ent_counts_bu += Counter(prop.arg_desc()[1] for prop in bin_Q_NE if prop.entity_types[1] == 'E')
	# ent_counts_b  = Counter(prop.arg_desc()[0] for prop in bin_Q_cand if prop.entity_types[0] == 'E')
	# ent_counts_b += Counter(prop.arg_desc()[1] for prop in bin_Q_cand if prop.entity_types[1] == 'E')
	ent_counts_b = Counter(tuple(sorted(prop.arg_desc())) for prop in bin_Q_NE)

	# Pick entities that have at least K mentions between unaries and binaries
	most_common_u = [e for e, count in (ent_counts_u + ent_counts_bu).most_common() if count >= reference.K_UNARY_ENT_MENTIONS]
	# most_common_u = [e for e, count in ent_counts_u.most_common() if count >= reference.K_UNARY_ENT_MENTIONS]
	most_common_b = [es for es, count in ent_counts_b.most_common() if count >= reference.K_BINARY_ENT_MENTIONS]

	top_most_common_u = most_common_u[:num_arts]
	top_most_common_b = most_common_b[:num_arts]

	# most_common_ents = most_common_u.union(most_common_b)

	########## SELECT CANDIDATE PROPOSITIONS

	# Generate questions from the most mentioned entities about the most popular predicates
	top_props_u = [p for p in un_Q_cand if p.arg_desc()[0] in top_most_common_u and p.pred in pred_cache]
	top_props_b = [p for p in bin_Q_cand if tuple(sorted(p.arg_desc())) in top_most_common_b and p.pred in pred_cache]

	# kept_top_props_u = [p for p in top_props_u if p.pred_desc() in uu_graphs[p.types[0]].nodes]
	# kept_top_props_b = [p for p in top_props_b if p.pred_desc() in bu_graphs['#'.join(p.basic_types)].nodes]

	sample_props = (top_props_u if 'unary' in question_modes else []) + \
				   (top_props_b if 'binary' in question_modes else [])
	# sample_props = kept_top_props_u + kept_top_props_b

	# Subsample propositions based on overall predicate frequency
	# top_props = [p for p in top_props if random.random() < reference.K_UNARY_PRED_MENTIONS / pred_cache[p.pred]]


	########## SAMPLE CANDIDATE PROPOSITIONS

	random.shuffle(sample_props)
	selected_questions = sample_props[:cap]

	# answer_choices = {q:{e for p in Q_cand for e in p.arg_desc() if e[e.index('#')+1:] == q[q.index('#')+1:]} for q in questions}
	# print('From {} sents > {} top u ents / {} top b ents; {} top u props / {} top b props; {} questions (cap {})'.format(num_sents, len(most_common_u), len(most_common_b), len(top_props_u), len(top_props_b), len(selected_questions), cap))
	return selected_questions, un_Q, bin_Q_all


def generate_top_pred_cache(articles: List[Article]) -> Dict[str, int]:
	pred_counter_u = Counter()
	pred_counter_b = Counter()
	for art in articles:
		pred_counter_u.update([p.pred for p in art.unary_props])
		pred_counter_b.update([p.pred for p in art.binary_props])

	top_pred_cache_u = {k: v for k, v in pred_counter_u.items() if v >= reference.K_UNARY_PRED_MENTIONS}
	top_pred_cache_b = {k: v for k, v in pred_counter_b.items() if v >= reference.K_BINARY_PRED_MENTIONS}

	top_pred_cache = {**top_pred_cache_u, **top_pred_cache_b}

	return top_pred_cache

def generate_question_partitions(articles: List[Article]) -> List[List[Article]]:
	# pred_counter = Counter()
	# for art in articles:
	# 	pred_counter.update([p.pred_desc() for p in art.unary_props])
	# top_preds = [(k, v) for k, v in pred_counter.most_common() if k.split('.')[0] not in reference.AUXILIARY_VERBS + reference.LIGHT_VERBS][:5000]
	# top_pred_cache = set([pred for pred, count in top_preds])

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

	return partitions


def generate_positive_question_sets(partitions: List[List[Article]],
									pred_cache: Dict[str, int],
									question_modes: Set[str],
									uu_graphs: Optional[EGraphCache],
									bu_graphs: Optional[EGraphCache]) -> \
		Tuple[List[List[Prop]], List[Tuple[List[Prop], List[Prop]]]]:

	P_list, evidence_list = [], []
	for partition in partitions:
		props, q_unaries, q_binaries = generate_questions(partition, pred_cache, question_modes, uu_graphs=uu_graphs, bu_graphs=bu_graphs)
		P_list.append(props)
		evidence_list.append((q_unaries, q_binaries))

	# plot_evidence_data(evidence_list)
	# exit(0)

	return P_list, evidence_list


# PLOT A BAR GRAPH OF ENTITY TYPES
def plot_evidence_data(evidence_list: List[Tuple[List[Prop], List[Prop]]], save:bool=True):
	un_Q_NE = [p for us, bs in evidence_list for p in us if p.entity_types[0] == 'E']
	bin_Q_NE = [p for us, bs in evidence_list for p in bs if 'E' in p.entity_types]

	unique_unary_args = list((p.args[0], p.basic_types[0]) for p in un_Q_NE)
	unique_binary_arg1 = list((p.args[0], p.basic_types[0]) for p in bin_Q_NE if p.entity_types[0] == 'E')
	unique_binary_arg2 = list((p.args[1], p.basic_types[1]) for p in bin_Q_NE if p.entity_types[1] == 'E')

	# Q_u_types = [p.types[0] for p in un_Q_NE]
	# Q_b_0_types = [p.basic_types[0] for p in bin_Q_NE if p.entity_types[0] == 'E']
	# Q_b_1_types = [p.basic_types[1] for p in bin_Q_NE if p.entity_types[1] == 'E']

	Q_u_types = [t for arg, t in unique_unary_args]
	Q_b1_types = [t for arg, t in unique_binary_arg1]
	Q_b2_types = [t for arg, t in unique_binary_arg2]

	all_types = set(Q_u_types + Q_b1_types + Q_b2_types)

	data = {'Arg Type': Q_u_types + Q_b1_types + Q_b2_types,
			'Prop Type': (['Unary'] * len(Q_u_types)) +
						 (['Binary Arg1'] * len(Q_b1_types)) +
						 (['Binary Arg2'] * len(Q_b2_types))}
	df = pd.DataFrame(data)

	order = [t for t, count in Counter(Q_u_types).most_common()]
	order += [t for t in all_types if t not in order]
	# print(order)

	plt.figure(figsize=(10, 10))
	ax = sns.countplot(y='Arg Type', data=df, hue='Prop Type', palette='colorblind', order=order)
	ax.set_xscale('log')
	# plt.xticks(rotation=80)
	plt.xlabel('Frequency')
	plt.ylabel('Category')
	plt.title('Frequency of Entity Types as Predicate Arguments')

	fname = 'type_chart_arg.png'
	if save:
		plt.savefig(fname)
		print('Type figure saved to', fname)
	else:
		plt.show()


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

def generate_tf_question_sets(articles: List[Article],
							  question_modes: Set[str],
							  negative_swaps: Optional[Dict[str, Dict[str, Any]]] = None,
							  uu_graphs: Optional[EGraphCache] = None,
							  bu_graphs: Optional[EGraphCache] = None,
							  filter_dict: Optional[Dict[str, Set[str]]] = None) -> \
		Tuple[List[List[Article]], List[List[Prop]], Optional[List[List[Prop]]], List[Tuple[List[Prop], List[Prop]]], Dict[str, int]]:
	print('Partitioning...', end=' ', flush=True)
	partitions = generate_question_partitions(articles)
	top_pred_cache = generate_top_pred_cache(articles)
	print('Generating positives...', end=' ', flush=True)
	P_list, evidence_list = generate_positive_question_sets(partitions, top_pred_cache, question_modes, uu_graphs, bu_graphs)
	print('Generating negatives...', end=' ', flush=True)
	N_list = None
	if negative_swaps and uu_graphs:
		N_list = generate_negative_question_sets(P_list, partitions, negative_swaps, uu_graphs, bu_graphs, filter_dict=filter_dict)

	print()
	return partitions, P_list, N_list, evidence_list, top_pred_cache

# generate negative questions for unaries only
# def generate_negative_question_sets(P_list: List[List[Prop]],
# 									partitions: List[List[Article]],
# 									negative_swaps: Dict[str, Dict[str, Any]],
# 									uu_graphs: Optional[EGraphCache],
# 									filter_dict: Optional[Dict[str, Set[str]]],
# 									cap: int) -> List[List[Prop]]:
# 	N_list = []
# 	num_antonyms = 0
# 	rej = defaultdict(Counter)
# 	swap_pairs = defaultdict(set)
#
#
# 	for i, ps in enumerate(P_list):
# 		# Cache the local predicates seen with each local entity
# 		ent_pred_mentions = defaultdict(set)
# 		for prop in sum((article.unary_props for article in partitions[i]), []):
# 			ent_pred_mentions[prop.arg_desc()[0]].add(prop.pred_desc())
# 		for prop in sum((article.binary_props for article in partitions[i]), []):
# 			# Cut binary into two "unaries"
# 			binary_pred_halves = prop.pred[prop.pred.find('(') + 1:prop.pred.find(')')].split(',')
# 			u0 = binary_pred_halves[0] + '#' + prop.basic_types[0]
# 			u1 = binary_pred_halves[1] + '#' + prop.basic_types[1]
# 			ent_pred_mentions[prop.arg_desc()[0]].add(u0)
# 			ent_pred_mentions[prop.arg_desc()[1]].add(u1)
#
# 		nprops = []
# 		max_per_positive = 5
#
# 		for j, positive_prop in enumerate(ps):
# 			if positive_prop.pred.startswith('be.'):
# 				continue
#
# 			ct = 0
#
# 			pred = positive_prop.pred
# 			typed_pred = positive_prop.pred_desc()
# 			pred_type = positive_prop.types[0]
# 			ent = positive_prop.arg_desc()[0]
#
# 			if uu_graphs and pred_type not in uu_graphs:
# 				# rej['no graph for predicate type'][pred_type] += 1
# 				continue
#
# 			if typed_pred not in uu_graphs[pred_type].nodes:
# 				# rej['pred not in graph'][typed_pred] += 1
# 				continue
#
# 			if pred not in negative_swaps:
# 				# rej['predicate has no swaps'][pred] += 1
# 				continue
#
# 			antonyms = negative_swaps[pred]['antonyms']
# 			troponyms = negative_swaps[pred]['troponyms']
# 			query_word = negative_swaps[pred]['query_word']
# 			# pred_relations = random.sample(antonyms, len(antonyms)) + random.sample(troponyms, len(troponyms))
# 			pred_relations = random.sample(troponyms, len(troponyms))
#
# 			# Filter potential relations
# 			if filter_dict and query_word in filter_dict:
# 				filter_out = filter_dict[query_word]
# 				pred_relations = [p for p in pred_relations if p not in filter_out]
#
# 			confirmed_swaps = []
# 			for a, relation in enumerate(pred_relations):
# 				# get swapped pred
# 				swapped_pred = Prop.swap_pred(typed_pred, relation)
#
# 				# is swap in same graph?
# 				if uu_graphs and swapped_pred not in uu_graphs[pred_type].nodes:
# 					rej['swapped pred is not in the graph'][pred + ' - ' + swapped_pred] += 1
# 					continue
#
# 				# Swap pair is testable
# 				swap_pairs[typed_pred].add(swapped_pred)
#
# 				# is swap mentioned at all?
# 				if swapped_pred in ent_pred_mentions[ent]:
# 					rej['swapped pred is actually mentioned'][swapped_pred] += 1
# 					continue
#
# 				# if known and not mentioned, add to list
# 				confirmed_swaps.append(Prop.from_descriptions(swapped_pred, [ent]))
# 				# if a < len(antonyms):
# 				# 	num_antonyms += 1
# 				ct += 1
# 				if ct >= max_per_positive:
# 					break
#
# 			if confirmed_swaps:
# 				nprops.extend(confirmed_swaps)
#
# 		random.shuffle(nprops)
# 		N_list.append(nprops[:(cap*len(partitions[i])*max_per_positive)])
#
# 	# print('Found {} antonyms and {} troponyms...'.format(num_antonyms, sum(len(n) for n in N_list)-num_antonyms), end=' ', flush=True)
# 	return N_list


def generate_negative_question_sets(P_list: List[List[Prop]],
									partitions: List[List[Article]],
									negative_swaps: Dict[str, Dict[str, Any]],
									uu_graphs: Optional[EGraphCache],
									bu_graphs: Optional[EGraphCache],
									filter_dict: Optional[Dict[str, Set[str]]]) -> List[List[Prop]]:
	N_list = []
	num_antonyms = 0
	rej = defaultdict(Counter)
	swap_pairs = defaultdict(set)

	article_pred_cache = {p.pred_desc() for articles in partitions for article in articles for p in article.unary_props + article.selected_binary_props}

	for i, ps in enumerate(P_list):
		# Cache the local predicates seen with each local entity
		ent_pred_mentions_u = defaultdict(set)
		ent_pred_mentions_b = defaultdict(set)
		for prop in sum((article.unary_props for article in partitions[i]), []):
			ent_pred_mentions_u[prop.arg_desc()[0]].add(prop.pred_desc())
		for prop in sum((article.binary_props for article in partitions[i]), []):
			# Cut binary into two "unaries"
			binary_pred_halves = prop.pred[prop.pred.find('(') + 1:prop.pred.find(')')].split(',')
			u0 = binary_pred_halves[0] + '#' + prop.basic_types[0]
			u1 = binary_pred_halves[1] + '#' + prop.basic_types[1]
			ent_pred_mentions_u[prop.arg_desc()[0]].add(u0)
			ent_pred_mentions_u[prop.arg_desc()[1]].add(u1)
			ent_pred_mentions_b[tuple(prop.arg_desc())].add(prop.pred_desc())

		nprops = []
		max_per_positive = 5

		for j, positive_prop in enumerate(ps):
			is_unary = len(positive_prop.args) == 1
			is_copula = positive_prop.pred.startswith('be.')

			ct = 0

			pred = positive_prop.pred
			query_word = proposition.extract_predicate_base_term(pred)
			typed_pred = positive_prop.pred_desc()
			pred_type = '#'.join(positive_prop.basic_types)

			# if is_unary and uu_graphs:
			# 	if pred_type not in uu_graphs or typed_pred not in uu_graphs[pred_type].nodes:
			# 		continue
			# elif not is_unary and bu_graphs:
			# 	if pred_type not in bu_graphs:
			# 		rej['no graph for predicate type'][pred_type] += 1
			# 		continue
			# 	elif typed_pred not in bu_graphs[pred_type].nodes:
			# 		rej['typed predicate not in graph'][typed_pred] += 1
			# 		continue
			# 	else:
			# 		rej['+ total typed binary predicates with available graph'][pred_type] += 1

			if query_word not in negative_swaps:
				if not is_unary:
					rej['predicate has no swaps'][pred] += 1
				continue

			if is_unary and is_copula:
				rels = negative_swaps[query_word]['hyponyms']
			else:
				antonyms = negative_swaps[query_word]['antonyms']
				troponyms = negative_swaps[query_word]['troponyms']
				# query_word = negative_swaps[pred]['query_word']
				# pred_relations = random.sample(antonyms, len(antonyms)) + random.sample(troponyms, len(troponyms))
				rels = troponyms

			pred_relations = random.sample(rels, len(rels))

			# Filter potential relations
			# if filter_dict and query_word in filter_dict:
			# 	filter_out = filter_dict[query_word]
			# 	pred_relations = [p for p in pred_relations if p not in filter_out]

			confirmed_swaps = []
			for a, relation in enumerate(pred_relations):
				# get swapped pred
				swapped_pred = Prop.swap_pred(typed_pred, relation)

				# is swap in same graph?
				# if is_unary and uu_graphs and swapped_pred not in uu_graphs[pred_type].nodes:
				# 	continue
				# elif not is_unary and bu_graphs and swapped_pred not in bu_graphs[pred_type].nodes:
				# 	rej['swapped predicate not in graph'][swapped_pred] += 1
				# 	continue
				if swapped_pred not in article_pred_cache:
					rej['swapped pred is never mentioned'][swapped_pred] += 1
					continue

				# Swap pair is testable
				swap_pairs[typed_pred].add(swapped_pred)

				# is swap mentioned at all?
				prop_args = positive_prop.arg_desc()
				if swapped_pred in ent_pred_mentions_u[prop_args[0]] or \
						swapped_pred in ent_pred_mentions_b[tuple(prop_args)]:
					if not is_unary:
						rej['swapped pred is actually mentioned in context'][swapped_pred] += 1
					continue

				# if known and not mentioned, add to list
				confirmed_swaps.append(Prop.from_descriptions(swapped_pred, prop_args))
				# if a < len(antonyms):
				# 	num_antonyms += 1
				ct += 1
				if ct >= max_per_positive:
					break

			if confirmed_swaps:
				nprops.extend(confirmed_swaps)

		random.shuffle(nprops)
		N_list.append(nprops[:len(ps)*max_per_positive])

	# print('Found {} antonyms and {} troponyms...'.format(num_antonyms, sum(len(n) for n in N_list)-num_antonyms), end=' ', flush=True)

	# print()
	# for key, counter in rej.items():
	# 	print(key, sum(v for k,v in counter.items()))
	# print()

	# sw = list(rej['swapped predicate not in graph'].items())
	# random.shuffle(sw)
	# print(sw[:100])
	# print()
	return N_list


# def generate_wh_question_sets(articles: List[Article]) -> \
# 		Tuple[List[List[str]], List[List[Set[str]]], List[Tuple[List[Prop], List[Prop]]]]:
# 	partitions, top_pred_cache = generate_question_partitions(articles)
# 	Q_list, A_list, _, evidence_list = generate_positive_question_sets(partitions, top_pred_cache, cap=3)
#
# 	return Q_list, A_list, evidence_list


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


def select_answerable_qs(P_list: List[List[Prop]], N_list: List[List[Prop]], uu_graphs: Optional[EGraphCache], bu_graphs: Optional[EGraphCache]) -> Tuple[List[List[Prop]], List[List[Prop]]]:
	def in_graphs(p: Prop):
		if len(p.args) == 1:
			t = p.types[0]
			if reference.GRAPH_BACKOFF:
				return prop_recognized_in_graphs(p, uu_graphs) or prop_recognized_in_graphs(p, bu_graphs)
			else:
				# return t in uu_graphs and p.pred_desc() in uu_graphs[t].nodes
				in_unary = t in uu_graphs and p.pred_desc() in uu_graphs[t].nodes
				in_binary = any(t in g and p.pred_desc() in bu_graphs[g].nodes for g in bu_graphs.keys())
				return in_unary or in_binary
		else:
			t = '#'.join(p.basic_types)
			if reference.GRAPH_BACKOFF:
				return prop_recognized_in_graphs(p, bu_graphs)
			else:
				return t in bu_graphs and p.pred_desc() in bu_graphs[t].nodes

	selected_P_list = [[p for p in ps if in_graphs(p)] for ps in P_list]
	selected_N_list = [[p for p in ps if in_graphs(p)] for ps in N_list]

	return selected_P_list, selected_N_list


# Input: list of partitioned positive questions, negative questions
# Input: desired percentage of questions to be unary (vs. binary), and percentage of positive (vs. negative)
def rebalance_qs(P_list: List[List[Prop]], N_list: List[List[Prop]], pct_unary:float=0.5, pct_pos:float=0.5) -> Tuple[List[List[Prop]], List[List[Prop]]]:
	# Adjust BINARY positivity ratio
	total_pos_b_qs = len([p for ps in P_list for p in ps if len(p.args) == 2])
	total_neg_b_qs = len([p for ps in N_list for p in ps if len(p.args) == 2])
	total_b_qs = total_pos_b_qs + total_neg_b_qs

	b_pos_rate = total_pos_b_qs / total_b_qs
	b_neg_rate = total_neg_b_qs / total_b_qs

	b_majority_class_pos = total_pos_b_qs > total_neg_b_qs
	if b_majority_class_pos:
		alpha = (pct_pos * b_neg_rate) / ((1 - pct_pos) * b_pos_rate)
		P_list = [[p for p in ps if len(p.args) == 1 or (len(p.args) == 2 and random.random() < alpha)] for ps in P_list]
	else:
		alpha = ((1 - pct_pos) * b_pos_rate) / (pct_pos * b_neg_rate)
		N_list = [[p for p in ps if len(p.args) == 1 or (len(p.args) == 2 and random.random() < alpha)] for ps in N_list]

	# Adjust UNARY positivity ratio
	total_pos_u_qs = len([p for ps in P_list for p in ps if len(p.args) == 1])
	total_neg_u_qs = len([p for ps in N_list for p in ps if len(p.args) == 1])
	total_u_qs = total_pos_u_qs + total_neg_u_qs

	u_pos_rate = total_pos_u_qs / total_u_qs
	u_neg_rate = total_neg_u_qs / total_u_qs

	u_majority_class_pos = total_pos_u_qs > total_neg_u_qs
	if u_majority_class_pos:
		alpha = (pct_pos * u_neg_rate) / ((1 - pct_pos) * u_pos_rate)
		P_list = [[p for p in ps if len(p.args) == 2 or (len(p.args) == 1 and random.random() < alpha)] for ps in
				  P_list]
	else:
		alpha = ((1 - pct_pos) * u_pos_rate) / (pct_pos * u_neg_rate)
		N_list = [[p for p in ps if len(p.args) == 2 or (len(p.args) == 1 and random.random() < alpha)] for ps in
				  N_list]

	# Adjust BINARY / UNARY ratio
	total_u_qs = len([p for ps in P_list + N_list for p in ps if len(p.args) == 1])
	total_b_qs = len([p for ps in P_list + N_list for p in ps if len(p.args) == 2])
	total_qs = total_u_qs + total_b_qs

	u_rate = total_u_qs / total_qs
	b_rate = total_b_qs / total_qs

	majority_class_u = total_u_qs > total_b_qs
	if majority_class_u:
		alpha = (pct_unary * b_rate) / ((1 - pct_unary) * u_rate)
		P_list = [[p for p in ps if len(p.args) == 2 or (len(p.args) == 1 and random.random() < alpha)] for ps in P_list]
		N_list = [[p for p in ps if len(p.args) == 2 or (len(p.args) == 1 and random.random() < alpha)] for ps in N_list]
	else:
		alpha = ((1 - pct_unary) * u_rate) / (pct_unary * b_rate)
		P_list = [[p for p in ps if len(p.args) == 1 or (len(p.args) == 2 and random.random() < alpha)] for ps in P_list]
		N_list = [[p for p in ps if len(p.args) == 1 or (len(p.args) == 2 and random.random() < alpha)] for ps in N_list]

	return P_list, N_list
