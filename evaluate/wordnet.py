import sys
import json
import re
import subprocess
from collections import defaultdict
from typing import *

def clean(pred: str) -> str:
	pred_str = pred.split('#')[0]
	word = pred_str.split('.')[0]
	return word

def process_infile(infile: str, outfile: str):
	with open(infile) as file:
		words = defaultdict(dict)
		backward_entailments = defaultdict(set)
		for pred in file:
			pred = pred.strip()
			word = clean(pred)

			trops = get_troponyms(word)
			forw_ents = get_entailments(word)
			ants = get_antonyms(word)
			# hyps = get_hypernyms(word)
			# for h in hyps:
			# 	ants.extend(get_antonyms(h))

			for cons in forw_ents:
				backward_entailments[cons].add(word)

			words[pred] = {'query_word': word,
						   'antonyms': ants,
						   'troponyms': trops,
						   'entails': forw_ents}

		for cons, ante_set in backward_entailments.items():
			if cons in words:
				words[cons]['entailed_by'] = list(ante_set)

	with open(outfile, 'w+') as file:
		json.dump(words, file, indent=2)


def get_antonyms(word: str) -> List[str]:
	return query_wordnet(word, 'antsv')

def get_troponyms(word: str) -> List[str]:
	return query_wordnet(word, 'hypov')

def get_hypernyms(word: str) -> List[str]:
	return query_wordnet(word, 'hypev')

def get_entailments(word: str) -> List[str]:
	return query_wordnet(word, 'entav')

def query_wordnet(word, mode) -> List[str]:
	query = '/usr/local/WordNet-3.0/bin/wn {} -{}'.format(word, mode)
	p = subprocess.Popen(query.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()
	if err:
		print(err)
		return []
	data = process_wn_out(out.decode('utf-8'))
	corrected_data = list(filter(lambda x: x != word, data))
	return corrected_data

def process_wn_out(output: str) -> List[str]:
	'''Sample output:
	Troponyms (hyponyms) of verb have

	12 of 19 senses of have

	Sense 1
	have, have got, hold
       => sustain, keep, maintain
       => keep, hold on
       => keep'''
	words = []
	# Accept only the first sense returned by WordNet (most commonly used)
	accepting_sense_data = True
	for line in output.split('\n'):
		if '=>' in line:
			accepting_sense_data = False
			term_block = line[line.find('=>')+2:].split(',')
			# Keep only one-word terms (no phrases)
			terms = [t.strip() for t in term_block if ' ' not in t.strip()]
			# Keep results in order while filtering out duplicates
			for t in terms:
				if t not in words:
					words.append(t)
		elif not accepting_sense_data:
			break
	return words

def main():
	if len(sys.argv) != 3:
		print('Usage: wordnet.py <infile> <outfile>')
		exit(1)

	print('Compiling WordNet data...')
	process_infile(sys.argv[1], sys.argv[2])
	print('done')

if __name__ == '__main__':
	main()