import sys
import json
import re
import subprocess
from collections import defaultdict
import utils
from typing import *

def clean(pred: str) -> str:
	pred_str = pred.split('#')[0]
	words = pred_str.split('.')
	if pred_str.startswith('be.'):
		word = '.'.join(words[:2])
	else:
		word = pred_str.split('.')[0]
	return word

def process_infile(infile: str, outfile: str, fallback_noun=False):
	with open(infile) as file:
		output = defaultdict(dict)
		backward_entailments = defaultdict(set)
		preds = file.readlines()
		words = list(set(clean(p.strip()) for p in preds))

		total = len(words)
		ct = 0
		print('Querying WordNet for {} words...'.format(total))
		for word in words:
			output[word] = {}

			if word.startswith('be.'):
				query = word[len('be.'):]
				hypos = get_hyponyms(query)
				hypo_preds = ['be.' + h for h in hypos]
				output[word]['hyponyms'] = hypo_preds

			else:
				trops = get_troponyms(word)
				forw_ents = get_entailments(word)
				ants = get_antonyms(word)
				hyps = get_hypernyms_verb(word)
				syns = get_synonyms(word)
				if not trops and not hyps and fallback_noun:
					trops = get_hyponyms(word)
					hyps = get_hypernyms_noun(word)
				# for h in hyps:
				# 	ants.extend(get_antonyms(h))

				for cons in forw_ents:
					backward_entailments[cons].add(word)

				output[word] = {'antonyms': ants,
								'troponyms': trops,
								'entails': forw_ents,
								'hypernyms': hyps,
								'synonyms': syns}
			ct += 1
			utils.print_progress(ct/total)

		print('Computing backward entailments...')
		for cons, ante_set in backward_entailments.items():
			if cons in words:
				output[cons]['entailed_by'] = list(ante_set)

	with open(outfile, 'w+') as file:
		json.dump(output, file, indent=2)


def get_antonyms(word: str) -> List[str]:
	return query_wordnet(word, 'antsv')

def get_troponyms(word: str) -> List[str]:
	return query_wordnet(word, 'hypov')

def get_hypernyms_verb(word: str) -> List[str]:
	return query_wordnet(word, 'hypev')

def get_hypernyms_noun(word: str) -> List[str]:
	return query_wordnet(word, 'hypen')

def get_entailments(word: str) -> List[str]:
	return query_wordnet(word, 'entav')

def get_hyponyms(word: str) -> List[str]:
	return query_wordnet(word, 'hypon')

def get_synonyms(word: str) -> List[str]:
	return query_wordnet(word, 'synsv')

def query_wordnet(word, mode) -> List[str]:
	query = '/usr/local/WordNet-3.0/bin/wn {} -{}'.format(word, mode)
	p = subprocess.Popen(query.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()
	if err:
		print(err)
		return []
	data = process_wn_out(out.decode('utf-8'), mode)
	corrected_data = list(filter(lambda x: x != word, data))
	return corrected_data

def process_wn_out(output: str, mode: str) -> List[str]:
	'''Sample output:
	Troponyms (hyponyms) of verb have

	12 of 19 senses of have

	Sense 1
	have, have got, hold
       => sustain, keep, maintain
       => keep, hold on
       => keep'''

	read_synset = mode == 'synsv'

	def extract_line(l):
		l = l.strip()
		if l.startswith('=>'):
			l = l[2:]
		raw_terms = [t.strip() for t in l.split(',')]

		# OLD: Keep only one-word terms (no phrases)
		# CURRENT: Keep all terms and join multiword exprs with '.'
		terms = ['.'.join(t.split()) for t in raw_terms]

		return terms

	words = []
	# Accept only the first sense returned by WordNet (most commonly used)
	lines = output.split('\n')
	i = 0
	while i < len(lines):
		line = lines[i]
		# For synonyms, capture the first synset, not the hypernyms also printed out
		if line.startswith('Sense') and read_synset:
			line = lines[i+1]
			words.extend(extract_line(line))
			break

		# Grab all first-level relations (not recursively higher) if we're scanning for a non-synonym relation:
		if line.strip().startswith('=>') and not read_synset:
			while line != '' and line.find('=>') < 10:
				words.extend(extract_line(line))
				i += 1
				line = lines[i]

			# Finish (keep only data from the first sense
			break
		i += 1

	unique_words = []
	for w in words:
		if w not in unique_words:
			unique_words.append(w)

	return unique_words

def main():
	# print('Running test...')
	# import pprint
	# pprint.pprint(query_wordnet('successes', 'hypon'))
	# pprint.pprint(query_wordnet('writer', 'hypov'))
	# exit(0)

	if len(sys.argv) < 3:
		print('Usage: wordnet.py <infile> <outfile> [--fallback-noun]')
		exit(1)

	print('Compiling WordNet data from {} ...'.format(sys.argv[1]))
	process_infile(sys.argv[1], sys.argv[2], fallback_noun=(len(sys.argv)==4))
	print('Done')

if __name__ == '__main__':
	main()