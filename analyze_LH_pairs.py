import argparse
import os

ARGS = None

def analyze(fname):
	print('Individual predicates found in training data')
	with open(fname) as file:
		ct = 0
		total = 0
		for line in file:
			if line == 'done':
				break
			if line.startswith('1'):
				ct += 1
			total += 1
		print('{} / {} ({:.2f}%)'.format(ct, total, ct/total*100))

	print()
	print('Edges found in training data')
	with open(fname) as file:
		ct = 0
		total = 0
		lines = file.readlines()
		for i in range(0, len(lines), 2):
			j = i+1
			if j >= len(lines) or lines[i] == 'done':
				break
			if lines[i].startswith('1') and lines[j].startswith('1'):
				ct += 1
			total += 1
		print('{} / {} ({:.2f}%)'.format(ct, total, ct/total*100))

	print()
	print('----------')


def main():
	global ARGS
	ARGS = parser.parse_args()
	analyze(ARGS.outfile)

parser = argparse.ArgumentParser(description='Analyze Levy/Holt dev relations found in train data')
parser.add_argument('outfile', help='Path to file of check-script output')

if __name__ == '__main__':
	main()
