import argparse
import os
import re
from datetime import datetime
from collections import Counter
from proposition import Prop
import json
from entailment import *
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
def scan_data_for_entities(fname_data: str, entities: Set[str]):
	print()

	found = set()
	with open(fname_data) as datafile:
		for i, line in enumerate(datafile):
			data = json.loads(line)
			for ew in data['ew']:
				wiki = ew['w']
				if wiki in entities:
					found.add(wiki)
			print('\r{} / {} entities found | {} lines read'.format(len(found), len(entities), i+1), end='')
	print()

def parse_val(val: str, val_type: str) -> datetime:
	templates = {
			'date': '%Y-%m-%d',
			'gYearMonth': '%Y-%m',
			'gYear': '%Y'
	}
	return datetime.strptime(val, templates[val_type])

# Read Yago db fact annotations
# <<	<http://yago-knowledge.org/resource/Libtiff>	<http://schema.org/copyrightHolder>	<http://yago-knowledge.org/resource/Silicon_Graphics>	>>	<http://schema.org/endDate>	"1997"^^<http://www.w3.org/2001/XMLSchema#gYear>	.
def get_annotations_from_file(fname_annots: str) -> List[Tuple[Tuple[str, str, str], datetime, str]]:
	annotations = []
	with open(fname_annots) as file:
		for line in file:
			r = r'<<(.*?)>>\s(\S*?)\s(\S*).*'
			m = re.match(r, line)
			if m:
				fact, ann, val = m.group(1, 2, 3)
				triple = tuple([read_uri_object(uri) for uri in fact.split('\t') if uri])
				assert len(triple) == 3
				annotation = read_uri_object(ann)

				val_str, val_type = read_valued_uri(val)
				value = parse_val(val_str, val_type)
				annotations.append((triple, value, annotation))

	return annotations


def main():
	global ARGS, MARGIN
	ARGS = parser.parse_args()

	print('Reading yago annotated facts from {} ...'.format(ARGS.yago_annotated))
	fact_annotations = get_annotations_from_file(ARGS.yago_annotated)
	print('Found {} date-annotated facts ({} start dates)'.format(len(fact_annotations), len([f for f in fact_annotations if f[2] == 'startDate'])))

	cutoff_date = datetime(2009,1,1)
	fact_annotations = [f for f in fact_annotations if cutoff_date < f[1]]
	print('Cut off facts by date')
	fact_ents = {e for trip, _, _ in fact_annotations for e in [trip[0], trip[2]]}
	print('{} involved entities'.format(len(fact_ents)))

	starts = Counter([r for (o, r, s), _, a in fact_annotations if a == 'startDate'])
	ends = Counter([r for (o, r, s), _, a in fact_annotations if a == 'endDate'])

	print('relations with startDate')
	for k,v in sorted(starts.items(), key=lambda x: x[1], reverse=True):
		print('{}:\t{}'.format(v, k))
	print()

	print('relations with endDate')
	for k, v in sorted(ends.items(), key=lambda x: x[1], reverse=True):
		print('{}:\t{}'.format(v, k))
	print()

	# print('Reading yago from {} ...'.format(ARGS.yago_facts))
	# fact_ents = get_fact_entities_from_file(ARGS.yago_facts)
	#
	print('Scanning data from {} for entities...'.format(ARGS.news_linked))
	scan_data_for_entities(ARGS.news_linked, fact_ents)

	print('Done')


parser = argparse.ArgumentParser(description='Yago compatibility with News Data')
parser.add_argument('yago_facts', help='Yago file containing facts')
parser.add_argument('news_linked', help='news file containing linked entities')
parser.add_argument('yago_annotated', help='Yago file containing fact annotations')


if __name__ == '__main__':
	main()