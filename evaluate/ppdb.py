from collections import defaultdict, Counter
import os
import sys
import proposition
import nltk
from typing import *

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

# def load_ppdb(fpath: str) -> Dict[str, Dict[str, Set[str]]]:
def load_ppdb(fpath: str) -> Dict[str, Set[str]]:
	file = next(f for f in os.listdir(fpath) if os.path.isfile(os.path.join(fpath, f)) and f.startswith('ppdb-'))
	# ppdb = defaultdict(lambda: defaultdict(set))
	ppdb = defaultdict(set)
	pos_types = Counter()

	with open(os.path.join(fpath, file)) as f:
		for line in f:
			parts = line.split('|||')

			pos = parts[0].strip()
			if not pos.startswith('[VB'):
				continue
			pos_types[pos] += 1

			kinds = ['Equivalence', 'ForwardEntailment', 'ReverseEntailment', 'Independent', 'OtherRelated']
			# features = {f[0]:float(f[1]) for f in [p.split('=') for p in parts[3].split()] if f[0] in kinds}
			# assert 0.95 < sum(features.values()) < 1.05
			# for k in kinds:
			# 	if k not in features:
			# 		features[k] = 0
			pp_kind = parts[-1].strip()
			if pp_kind not in ['Equivalence', 'ForwardEntailment', 'OtherRelated']:
				continue
			# if features['Equivalence'] < 0.2 or features['ForwardEntailment'] < 0.2:
			# 	continue

			w1 = parts[1].strip()
			w2 = parts[2].strip()
			l1 = lemmatizer.lemmatize(w1, 'v')
			l2 = lemmatizer.lemmatize(w2, 'v')

			# ppdb[l1][pp_kind].add(l2)
			ppdb[l1].add(l2)
			# ppdb[w1].add(w2)

	return ppdb


def main():
	data_folder = sys.argv[1]
	wordnet = proposition.read_substitution_pairs(os.path.join(data_folder, 'substitution_pairs_person.json'))
	filtered = proposition.read_substitution_pairs(os.path.join(data_folder, 'substitution_pairs_person_filtered.json'))
	ppdb = load_ppdb(data_folder)

	total = 0
	missing_p = 0
	agree_keep = 0
	agree_remove = 0
	para_only = 0
	filter_only = 0
	ppdb_hits = set()

	for pred, data in wordnet.items():
		query_word = data['query_word']
		for trop in data['troponyms']:
			total += 1
			filtered_out = pred not in filtered or trop not in filtered[pred]['troponyms']

			if query_word not in ppdb:
				missing_p += 1
				continue

			ppdb_hits.add(query_word)
			# para = trop in ppdb[query_word]['Equivalence']
			para = trop in ppdb[query_word]

			if filtered_out and para:
				agree_remove += 1
			elif not filtered_out and not para:
				agree_keep += 1
			elif para and not filtered_out:
				para_only += 1
				print('para only: {} - {}'.format(query_word, trop))
			elif filtered_out and not para:
				filter_only += 1

	all_comparisons = agree_keep + agree_remove + para_only + filter_only

	print('Agree: {:.2f}, {}/{} compared'.format((agree_keep + agree_remove)/all_comparisons, all_comparisons, total))
	print('Done')


if __name__ == '__main__':
	main()