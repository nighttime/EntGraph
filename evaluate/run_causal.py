import argparse
import os
import re
from datetime import datetime
from collections import Counter
import json
from dataclasses import dataclass
import pickle

import utils
from reference import tcolors
from proposition import Prop
from article import *
from entailment import *
# import pdb

from typing import *

# <http://yago-knowledge.org/resource/Arundhati_(2014_film)>
def read_uri_object(uri: str) -> str:
	path_entity = uri.split('/')[-1][:-1]
	# entity_name = ' '.join(path_entity.split('_'))
	# lowered = entity_name.lower()
	return path_entity

# "1997"^^<http://www.w3.org/2001/XMLSchema#gYear>
def read_valued_uri(valued_uri: str) -> Tuple[str, str]:
	value, uri = valued_uri.split('^^')
	value = value[1:-1]
	path_entity = read_uri_object(uri)
	entity_type = path_entity.split('#')[1]
	# entity_name = ' '.join(path_entity.split('_'))
	# lowered = entity_name.lower()
	return value, entity_type

# Read Yago db file and cache entity names
def get_fact_entities_from_file(fname_yago_facts: str) -> Set[str]:
	entities = set()
	with open(fname_yago_facts) as file:
		for line in file:
			parts = line.split('\t')
			ob, rel, sub = parts[:3]
			ob = read_uri_object(ob)
			sub = read_uri_object(sub)
			entities.add(ob)
			entities.add(sub)
	return entities

# Scan through NewsCrawl linked file and
def scan_data_for_entities(fname_data: str, entities: Set[Tuple[str]]):
	print()

	pairs_by_ent = defaultdict(set)
	for e1, e2 in entities:
		pairs_by_ent[e1].add(e2)
		pairs_by_ent[e2].add(e1)

	found = set()
	with open(fname_data) as datafile:
		for i, line in enumerate(datafile):
			data = json.loads(line)
			wikis = {ew['w'].lower() for ew in data['ew']}
			# for pair in entities:
			# 	if all(e in wikis for e in pair):
			# 		found.add(pair)

			for w1 in wikis:
				for w2 in pairs_by_ent[w1]:
					if w2 in wikis:
						found.add(tuple(sorted([w1,w2])))

			if i % 100 == 0:
				print('\r{} / {} entity pairs found | {} lines read'.format(len(found), len(entities), i+1), end='')
	print()

def parse_val(val: str, val_type: str) -> datetime:
	templates = {
			'date': '%Y-%m-%d',
			'gYearMonth': '%Y-%m',
			'gYear': '%Y'
	}
	return datetime.strptime(val, templates[val_type])

@dataclass
class AnnotedTriple:
	triple: Tuple[str, str, str]
	date: datetime
	label: str

# Read Yago db fact annotations
# <<	<http://yago-knowledge.org/resource/Libtiff>	<http://schema.org/copyrightHolder>	<http://yago-knowledge.org/resource/Silicon_Graphics>	>>	<http://schema.org/endDate>	"1997"^^<http://www.w3.org/2001/XMLSchema#gYear>	.
def get_annotations_from_file(fname_annots: str) -> List[AnnotedTriple]:
	annotations = []
	with open(fname_annots) as file:
		for line in file:
			r = r'<<(.*?)>>\s(\S*?)\s(\S*).*'
			m = re.match(r, line)
			if m:
				fact, ann, val = m.group(1, 2, 3)
				triple_parts = [read_uri_object(uri) for uri in fact.split('\t') if uri]
				assert len(triple_parts) == 3
				triple = (triple_parts[0], triple_parts[1], triple_parts[2])
				annotation = read_uri_object(ann)

				val_str, val_type = read_valued_uri(val)
				value = parse_val(val_str, val_type)
				annotations.append(AnnotedTriple(triple, value, annotation))

	return annotations

def filter_facts(facts: List[AnnotedTriple]) -> List[AnnotedTriple]:
	cutoff_date = datetime(2008, 1, 1)
	# facts = [f for f in facts if f.label == 'startDate']
	# start_facts = [f for f in facts if f.label == 'endDate']
	cutoff_facts = [f for f in facts if cutoff_date < f.date]
	print('Found {} date-annotated facts > {} after cutoff date'.format(len(facts), len(cutoff_facts)))
	return cutoff_facts

def binary_prop_from_fact(fact: AnnotedTriple) -> Optional[Prop]:
	mapping_forwards = {
			'subOrganization': '(own.1,own.2)',
			'location': '(at.1,at.2)',
			'spouse': '(marry.2,marry.to.2)',
			# 'alumniOf': '(receive.1,receive.degree.from.2)',
			# 'containedInPlace': '(in.1,in.2)',
	}
	mapping_backwards = {
			'parentOrganization': '(own.1,own.2)',
			# 'containsPlace': '(in.1,in.2)',
	}

	db_relation = fact.triple[1]
	if db_relation in mapping_forwards:
		pred_untyped = mapping_forwards[db_relation]
		backwards_relation = False
	elif db_relation in mapping_backwards:
		pred_untyped = mapping_backwards[db_relation]
		backwards_relation = True
	else:
		return None

	args = [fact.triple[0].lower(), fact.triple[2].lower()]
	typing_1 = get_type(args[0], 'E')
	typing_2 = get_type(args[1], 'E')
	symmetric_typing = typing_1 == typing_2

	if symmetric_typing:
		typing_1, typing_2 = typing_1 + '_1', typing_2 + '_2'
		if backwards_relation: # FLIP!
			args = [args[1], args[0]]
	elif backwards_relation: #FLIP!
		typing_1, typing_2 = typing_2, typing_1
		args = [args[1], args[0]]

	# make prop...
	# pred = pred_untyped + '#' + typing_1 + '#' + typing_2
	prop = Prop(pred_untyped, args, date=fact.date)
	prop.set_types([typing_1, typing_2])

	return prop



def run_kb_prediction(Q: List[Prop], A: List[AnnotedTriple], articles: List[Article], model: EGraphCache, edge_typing=True) -> List[Optional[datetime]]:
	fact_index: Dict[Tuple[str, str], List[Prop]] = {}
	for art in articles:
		for ev in art.binary_props:
			if 'E' in ev.entity_types:
				arg_desc = ev.arg_desc()
				key = (arg_desc[0], arg_desc[1])
				if key not in fact_index:
					fact_index[key] = []
				fact_index[key].append(ev)

	answer_dates = []

	# Loop over all questions
	missing_graphs = Counter()
	ds_hits = 0
	for i,q in enumerate(Q):
		if q.type_desc() not in model:
			answer_dates.append(None)
			missing_graphs[q.type_desc()] += 1
			continue
		ants = model[q.type_desc()].get_antecedents(q.pred_desc())
		# ants = [a for ant_set in query_all_graphs_for_prop(q, model) for a in ant_set]
		if edge_typing:
			ants = [e for e in ants if e.edge_type == EdgeType.CONSEQUENCE_SUCCESS]
		arg_desc = tuple(q.arg_desc())
		evidence = fact_index.get((arg_desc[0], arg_desc[1]), []) + fact_index.get((arg_desc[1], arg_desc[0]), [])
		if evidence:
			ds_hits += 1

		best_date = datetime.now()
		best_ev = None
		found = False
		for ev in evidence:
			for a in ants:
				if ev.pred_desc() == a.pred and ev.date < best_date:
					best_ev = ev
					best_date = ev.date
					found = True

		correct = A[i].date.year == best_date.year
		if found:
			if correct:
				print(tcolors.OKGREEN + '{} -> {} [{}]'.format(best_ev, q, best_date.year) + tcolors.ENDC)
			else:
				print(tcolors.FAIL + '{} -> {} [c{} | p{}]'.format(best_ev, q, A[i].date.year, best_date.year) + tcolors.ENDC)

		elif evidence:
			print('No entailments found for: {} in {} evidence props: {}'.format(q.pred_desc(), len(evidence), evidence))

		answer_dates.append(best_date if found else None)

	print(tcolors.OKGREEN + 'hits in dataset: {} / {}'.format(ds_hits, len(Q)) + tcolors.ENDC)
	print(tcolors.WARNING + 'No graph found for typing: {}'.format(missing_graphs.most_common()) + tcolors.ENDC)
	return answer_dates


def score_preds(A: List[AnnotedTriple], A_pred: List[Optional[datetime]]) -> List[Optional[int]]:
	assert len(A) == len(A_pred)
	scores =  []
	for a, a_pred in zip(A, A_pred):
		if a_pred:
			score = 1 if a.date.year == a_pred.year else 0
			scores.append(score)
		else:
			scores.append(None)

	return scores

def calc_stats(marked: List[Optional[int]]) -> Tuple[float, float]:
	tp = sum(1 for m in marked if m is not None and m == 1)
	fp = sum(1 for m in marked if m is not None and m == 0)
	fn = sum(1 for m in marked if m is None)
	if tp + fp == 0:
		return 0, 0
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	return precision, recall

def main():
	global ARGS, MARGIN
	ARGS = parser.parse_args()

	utils.print_BAR()
	print('Causal QA')
	utils.checkpoint()
	utils.print_bar()

	print('Reading entity type data from {} ...'.format(ARGS.data_folder))
	load_precomputed_entity_types(ARGS.data_folder)

	yago_folder = os.path.join(ARGS.data_folder, 'yago')
	# yago_facts_fname = os.path.join(yago_folder, 'yago-wd-facts_10k.nt')
	yago_annotated_fname = os.path.join(yago_folder, 'yago-wd-annotated-facts.ntx')

	print('Reading yago annotated facts from {} ...'.format(yago_annotated_fname))
	fact_annotations = get_annotations_from_file(yago_annotated_fname)
	facts = filter_facts(fact_annotations)
	Q, A = [], []
	for f in facts:
		q = binary_prop_from_fact(f)
		if q:
			Q.append(q)
			A.append(f)

	queried_entities = {tuple(sorted(q.args)) for q in Q}

	utils.checkpoint()
	utils.print_bar()

	if ARGS.prescan_news_linked:
		# print('Scanning data from {} for entities...'.format(ARGS.news_linked))
		scan_data_for_entities(ARGS.prescan_news_linked, queried_entities)

	if ARGS.no_eval:
		return

	fpath_news_data = os.path.join(ARGS.data_folder, 'news_data_yago.pkl')
	if ARGS.read_news:
		print('Reading precomputed source articles from {} ...'.format(fpath_news_data))
		with open(fpath_news_data, 'rb') as f:
			articles = pickle.load(f)
	else:
		print('Reading source articles from {} ...'.format(ARGS.news_gen_file))
		articles, _, _ = read_source_data(ARGS.news_gen_file, save_sents=False, read_dates=True, target_entities=queried_entities)
	if ARGS.save_news:
		with open(fpath_news_data, 'wb+') as f:
			pickle.dump(articles, f, pickle.HIGHEST_PROTOCOL)

	print('Reading CG files from {} ...'.format(ARGS.cgraphs))
	cg_model = read_precomputed_EGs(ARGS.cgraphs)
	print('Reading EG files from {} ...'.format(ARGS.egraphs))
	eg_model = read_precomputed_EGs(ARGS.egraphs)

	utils.checkpoint()
	utils.print_bar()

	# pdb.set_trace()

	print('Causal Model')
	predictions = run_kb_prediction(Q, A, articles, cg_model, edge_typing=True)
	marked = score_preds(A, predictions)
	p, r = calc_stats(marked)
	print('predictions: {}\nprecision: {:.2f}\nrecall: {:.2f}'.format(sum(1 for m in marked if m is not None), p, r))
	utils.checkpoint()
	utils.print_bar()

	# print('Entailment Model')
	# predictions = run_kb_prediction(Q, A, articles, eg_model, edge_typing=False)
	# marked = score_preds(A, predictions)
	# p, r = calc_stats(marked)
	# print('predictions: {}\nprecision: {:.2f}\nrecall: {:.2f}'.format(sum(1 for m in marked if m is not None), p, r))
	# utils.print_bar()

	# News_Linked
	# ../news_gen/newscrawl/newscrawl_negation_selection_2020_5_19/newsC_linked_10k.json

	# fact_annotations = [f for f in fact_annotations if cutoff_date < f[1]]
	# print('Cut off facts by date')
	# fact_ents = {e for trip, _, _ in fact_annotations for e in [trip[0], trip[2]]}
	# print('{} involved entities'.format(len(fact_ents)))

	# starts = Counter([r for (o, r, s), _, a in fact_annotations if a == 'startDate'])
	# ends = Counter([r for (o, r, s), _, a in fact_annotations if a == 'endDate'])

	# print('relations with startDate')
	# for k,v in sorted(starts.items(), key=lambda x: x[1], reverse=True):
	# 	print('{}:\t{}'.format(v, k))
	# print()

	# print('relations with endDate')
	# for k, v in sorted(ends.items(), key=lambda x: x[1], reverse=True):
	# 	print('{}:\t{}'.format(v, k))
	# print()

	# print('Reading yago from {} ...'.format(ARGS.yago_facts))
	# fact_ents = get_fact_entities_from_file(ARGS.yago_facts)
	#


	print('Done')


parser = argparse.ArgumentParser(description='Resolving Yago times with News Data')
parser.add_argument('news_gen_file', help='Path to file used for partition into Question set and Answer set')
parser.add_argument('data_folder', help='Path to data folder including freebase entity types and predicate substitution pairs')
# parser.add_argument('yago_annotated', help='Yago file containing fact annotations')
parser.add_argument('cgraphs', help='Path to causal graph zip for use in question answering')
parser.add_argument('egraphs', help='Path to entailment graph zip for use in question answering')
# parser.add_argument('--text-graphs', action='store_true', help='Flag if graphs are to be read in from raw text files')
parser.add_argument('--save-news', action='store_true', help='Save the news file after reading it in and processing it for later use')
parser.add_argument('--read-news', action='store_true', help='Read the news file in (after saving it) for use now')
# parser.add_argument('graph_folder', help='Path to causal graph folder to assist question answering')
# parser.add_argument('yago_facts', help='Yago file containing facts')
parser.add_argument('--prescan-news-linked', action='store_true', help='news file containing linked entities')
parser.add_argument('--no-eval', action='store_true', help='Quit before reading in model or news data, or evaluating')

if __name__ == '__main__':
	main()