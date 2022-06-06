import json
import re
import gc
from collections import defaultdict, Counter
import random
from datetime import datetime
import argparse

import utils
from proposition import *
from collections import Counter
from typing import *
import random

REQUIRE_ONE_NE = True

class Article:
	def __init__(self, art_ID, date):
		self.art_ID = art_ID
		self.date = date
		self.unary_props: List[Prop] = []
		self.binary_props: List[Prop] = []
		self.selected_binary_props: List[Prop] = []
		self.sents = []

	def add_unary(self, unary):
		self.unary_props.append(unary)

	def add_binary(self, binary):
		self.binary_props.append(binary)

	def add_selected_binary(self, binary):
		self.selected_binary_props.append(binary)

	def remove_qa_pair(self, q, a):
		for i,u in enumerate(self.unary_props):
			if u.pred_desc() == q and u.args[0] == a:
				self.unary_props.pop(i)
				break

	def arg_mentions(self, arg: str) -> int:
		return len([p for p in self.unary_props + self.binary_props if arg in p.args])

	def feature_counts(self) -> Counter[str]:
		return Counter(self.named_entity_mentions())

	def feature_set(self) -> Set[str]:
		return set(self.feature_counts().keys())

	def named_entity_mentions(self, typed=True) -> List[str]:
		ents = []
		for p in self.unary_props + self.binary_props:
			for i, e_type in enumerate(p.entity_types):
				if e_type == 'E':
					ents.append(p.arg_desc()[i] if typed else p.args[i])
		return ents

	def remove_uprops(self, props):
		for p in props:
			self.unary_props.remove(p)

simples = [re.compile(s) for s in [r'be\.\d', r'do\.\d']]

def reject_unary(pred: str) -> bool:
	if '#' in pred:
		pred = pred.split('#')[0]

	if any(c in pred for c in ['\'', '`']):
		return True

	if any(pred.startswith(v + '.') for v in reference.AUXILIARY_VERBS):
		return True

	global simples
	if any(r.match(pred) for r in simples):
		return True

	if any(pred.startswith(v + '.') for v in reference.REPORTING_VERBS):
		return True

	if any(pred.startswith(v + '.') for v in reference.PREPOSITIONS):
		return True

	if '.e.' in pred:
		return True

	return False

def reject_binary(pred: str) -> bool:
	# if pred.find('(') != 0 or pred.count(',') != 1: #changed on Mar 11 2021
	if pred.find('(') == -1 or pred.count(',') != 1:
		return True

	pred = pred[pred.find('(')+1:pred.find(')')]
	parts = pred.split(',')

	# filter out malformed extractions e.g. (1,with.2)#person#thing
	if any(len(p.split('.')) < 2 for p in parts):
		return True

	# filter out malformed extractions e.g. (``, ...
	if any(any(c in p for c in ['\'', '`']) for p in parts):
		return True

	if any(any(p.startswith(v + '.') for v in reference.AUXILIARY_VERBS) for p in parts):
		return True

	global simples
	if all(any(r.match(p) for r in simples) for p in parts):
		return True

	return False

# ((of.1,of.2)::first::the year::EE::0::4::null::null::Tue_Jan_01_00:00:00_GMT_2013::Tue_Dec_31_00:00:00_GMT_2013)
def read_source_data(source_fname: str, save_sents=True, read_dates=False, target_entities:Optional[Set[str]]=None) -> Tuple[List[Article], List[Prop], List[Prop]]:
	global REQUIRE_ONE_NE
	articles = {}
	unary_props = []
	binary_props = []
	with open(source_fname) as source_data:
		gc_collect = False
		for linenum,line in enumerate(source_data):
			if not line.startswith('{'):
				continue
			l = json.loads(line)
			try:
				art_ID = int(l['articleId'])
			except:
				continue
			if art_ID not in articles:
				date = datetime.strptime(l['date'], '%b %d, %Y %X %p')
				articles[art_ID] = Article(art_ID, date)

			line_ID = l['lineId']
			sent = l['s']
			if save_sents:
				articles[art_ID].sents.append((line_ID, sent))

			unaries_raw = [u['r'][1:-1] for u in l['rels_unary']]
			for i, u in enumerate(unaries_raw):
				parts = u.split('::')

				norm_pred, _ = normalize_predicate(parts[0])
				if reject_unary(norm_pred):
					continue

				norm_ent = normalize_entity(parts[1])
				if target_entities and norm_ent not in target_entities:
					continue
				entity_type = parts[2]
				if REQUIRE_ONE_NE and 'E' not in entity_type:
					continue
				typing = get_type(norm_ent, 'E' in entity_type)

				prop = Prop(norm_pred, [norm_ent])
				prop.set_entity_types(entity_type)
				prop.set_types([typing])

				unary_props.append(prop)
				articles[art_ID].add_unary(prop)

			binaries_raw = [u['r'][1:-1] for u in l['rels']]
			for i, b in enumerate(binaries_raw):
				parts = b.split('::')

				norm_pred, reversed_pred = normalize_predicate(parts[0])
				if reject_binary(norm_pred):
					continue

				norm_ents = [normalize_entity(e) for e in parts[1:3]]
				if target_entities and tuple(sorted(norm_ents)) not in target_entities:
					continue
				entity_types = parts[3]
				if not all(e in 'GE' for e in entity_types):
					continue
				# if REQUIRE_ONE_NE and 'E' not in entity_types:
				# 	continue
				typing = [get_type(norm_ents[i], 'E' == entity_types[i]) for i in range(2)]

				prop_date = None
				if read_dates:
					start_date = parts[-2]
					if len(parts) < 10 or start_date == 'null':
						prop_date = articles[art_ID].date
					else:
						date_parts = start_date.split('_')
						date_str = '-'.join([date_parts[i] for i in [1,2,5]])
						try:
							prop_date = datetime.strptime(date_str, '%b-%d-%Y')
						except:
							try:
								prop_date = datetime.strptime(date_str, '%b-%d-%y')
							except:
								prop_date = articles[art_ID].date

				if reversed_pred:
					norm_ents = list(reversed(norm_ents))
					entity_types = entity_types[::-1]
					typing = list(reversed(typing))

				type_symmetric = typing[0] == typing[1]
				if type_symmetric:
					typing = [typing[0] + '_1', typing[1] + '_2']

				prop = Prop(norm_pred, norm_ents, date=prop_date)
				prop.set_entity_types(entity_types)
				prop.set_types(typing)
				binary_props.append(prop)
				articles[art_ID].add_binary(prop)
				articles[art_ID].add_selected_binary(prop)

				if type_symmetric:
					norm_ents = list(reversed(norm_ents))
					entity_types = entity_types[::-1]
					typing = list(reversed(typing))
					prop_rev = Prop(norm_pred, norm_ents, date=prop_date)
					prop_rev.set_entity_types(entity_types)
					prop_rev.set_types(typing)
					binary_props.append(prop_rev)
					articles[art_ID].add_binary(prop_rev)

			if linenum % 10000 == 0:
				if linenum > 0 and linenum % 3000000 == 0:
					gc_collect = True

				print('\r{} lines read {}'.format(linenum, '(GC Collecting!)' if gc_collect else ' '*20), end='', flush=True)

				if gc_collect:
					gc.collect()
					gc_collect = False


	print()

	arts = [a for artID, a in articles.items()]

	# for art in arts:
	# 	start_size = len(art.selected_binary_props)
	# 	art.selected_binary_props = [p for p in art.selected_binary_props if not p.types[0].endswith('_2')]
	# 	end_size = len(art.selected_binary_props)
	# 	if start_size != end_size:
	# 		print('removed {} props ({:.2f}%)'.format(start_size - end_size, (start_size - end_size)/start_size))

	for art in arts:
		art.sents = [s for line_ID, s in sorted(art.sents)]

	return arts, unary_props, binary_props


def main():
	global ARGS
	ARGS = parser.parse_args()
	print('Reading entity type data from {} ...'.format(ARGS.data_folder))
	load_precomputed_entity_types(ARGS.data_folder)

	print('Reading source articles from {} ...'.format(ARGS.news_gen_file))
	articles, _, _ = read_source_data(ARGS.news_gen_file)

	random.shuffle(articles)
	sample = articles[:10]

	with open('article_sample.txt', 'w+') as f:
		for a in sample:
			f.write(str(a.art_ID) + '\n')
			f.write(str(a.date) + '\n')
			f.writelines(a.sents)
			f.write('\n')
			f.writelines([str(p) for p in a.selected_binary_props])
			f.write('\n\n\n')

parser = argparse.ArgumentParser(description='Print out sample of articles')
parser.add_argument('news_gen_file', help='Path to file used for partition into Question set and Answer set')
parser.add_argument('data_folder', help='Path to data folder including freebase entity types and predicate substitution pairs')

if __name__ == '__main__':
	main()