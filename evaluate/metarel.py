import argparse
import os
from proposition import Prop
from collections import defaultdict
from entailment import Entailment, EGraphCache, EGStage, EntailmentGraph, read_graphs, read_precomputed_EGs
from typing import *

MARGIN = 0.3

KEY_CONS_SUCC = 'consequences_success'
KEY_PRECOND = 'preconditions'
KEY_CONS_FAIL = 'consequences_failure'

# def filter_keep_pos_only(pos_ents: List[entailment.Entailment], neg_ents: List[entailment.Entailment]) -> List[entailment.Entailment]:
# 	neg_map = {e.pred:e.score for e in neg_ents}
# 	kept_ents = []
# 	global ARGS
# 	for ent in pos_ents:
# 		if ent.score < 0.01:
# 			continue
#
# 		if ARGS.decision == 'strict':
# 			if ent.pred in neg_map:
# 				continue
# 		elif ARGS.decision == 'gt':
# 			if ent.pred in neg_map and ent.score <= neg_map[ent.pred]:
# 				continue
# 		elif ARGS.decision == 'margin':
# 			if ent.pred in neg_map:
# 				neg_score = neg_map[ent.pred]
# 				if ent.score < neg_score or (ent.score - neg_score) / ent.score < MARGIN:
# 					continue
# 		kept_ents.append(ent)
#
# 	return kept_ents

# def filter_keep_pos_intersect_neg(pos_ents: List[entailment.Entailment], neg_ents: List[entailment.Entailment]) -> List[entailment.Entailment]:
# 	neg_map = {e.pred:e.score for e in neg_ents}
# 	kept_ents = []
# 	for ent in pos_ents:
# 		if ent.score < 0.01:
# 			continue
#
# 		if ARGS.decision == 'strict':
# 			if ent.pred not in neg_map:
# 				continue
# 		elif ARGS.decision == 'gt':
# 			raise Exception('gt not supported for intersect')
# 		elif ARGS.decision == 'margin':
# 			if ent.pred in neg_map:
# 				neg_score = neg_map[ent.pred]
# 				if abs(ent.score - neg_score) / ent.score > MARGIN:
# 					continue
# 			else:
# 				continue
# 		kept_ents.append(ent)
#
# 	return kept_ents

# def filter_keep_neg_only(pos_ents: List[Entailment], neg_ents: List[Entailment]) -> List[Entailment]:
# 	pos_preds = set([e.pred for e in pos_ents])
# 	kept_ents = []
# 	for ent in neg_ents:
# 		if ent.pred in pos_preds:
# 			continue
# 		kept_ents.append(ent)
#
# 	return kept_ents

CompareFun = Callable[[List[Entailment], List[Entailment]], List[Entailment]]
# FUN_MAP = {'pos': filter_keep_pos_only, 'inter': filter_keep_pos_intersect_neg, 'neg': filter_keep_neg_only}

def analyze_causal(graphs: EGraphCache, compare_fun: CompareFun):
	global ARGS

	if not os.path.exists(ARGS.outdir):
		os.makedirs(ARGS.outdir)

	seen_graphs = set()
	for typing, graph in graphs.items():
		if graph in seen_graphs:
			continue

		seen_graphs.add(graph)
		new_edges = {}

		# Analyze graph
		for neg_pred in graph.nodes:
			if not neg_pred.startswith('NEG__'):
				continue

			pos_pred = neg_pred[len('NEG__'):]
			if pos_pred not in graph.nodes:
				continue

			pos_ents = graph.get_entailments(pos_pred)
			neg_ents = graph.get_entailments(neg_pred)

			net_ents = compare_fun(pos_ents, neg_ents)
			if net_ents:
				new_edges[pos_pred] = net_ents

		if new_edges:
			new_graph = EntailmentGraph.from_edges(new_edges, typing, graph.space, graph.stage, keep_forward=True)
			file_suffix = '_sim.txt' if graph.stage == EGStage.LOCAL else '_binc.txt'
			fpath = os.path.join(ARGS.outdir, graph.typing + file_suffix)
			new_graph.write_to_file(fpath)


def compare_entailments(pos_ents: List[Entailment], neg_ents: List[Entailment]) -> Tuple[List[Entailment], List[Entailment], List[Entailment]]:
	global ARGS

	pos_ents = [e for e in pos_ents if e.score >= 0.01]
	neg_ents = [e for e in neg_ents if e.score >= 0.01]

	neg_map = {e.pred: e.score for e in neg_ents}
	intersection, positives, negatives = [], [], []

	# Select intersection based on decision function
	for ent in pos_ents:
		if ARGS.decision == 'strict':
			# if ent.pred in neg_map:
			# 	intersection.append(ent)
			raise Exception('NYI: strict decision function')
		elif ARGS.decision == 'gt':
			raise Exception('NYI: greater-than decision function')
		elif ARGS.decision == 'margin':
			if ent.pred in neg_map:
				neg_score = neg_map[ent.pred]
				if abs(ent.score - neg_score) / ent.score <= MARGIN:
					intersection.append(ent)
				elif ent.score > neg_score:
					positives.append(ent)
				else:
					negatives.append(ent)
			else:
				positives.append(ent)

	used = {e.pred for e in intersection + positives + negatives}
	for ent in neg_ents:
		if ent.pred not in used:
			negatives.append(ent)

	positives.sort(key=lambda x: x.score, reverse=True)
	intersection.sort(key=lambda x: x.score, reverse=True)
	negatives.sort(key=lambda x: x.score, reverse=True)

	# Consistency check
	all_inputs = [e.pred for e in pos_ents + neg_ents]
	all_outputs = [e.pred for e in positives + intersection + negatives]
	# try:
	# 	assert len(all_outputs) == len(set(all_outputs))
	# except:
	# 	print('len(all_outputs) = {} ; len(set(all_outputs)) = {}'.format(len(all_outputs), len(set(all_outputs))))
	# 	missing = [p for p in all_outputs if all_outputs.count(p) > 1]
	# 	print('extra: ' + str(missing))
	# 	print()
	assert all(pred in all_outputs for pred in all_inputs)

	return positives, intersection, negatives

def write_causal_graph(new_edges: Dict[str, Dict[str, List[Entailment]]], stage: EGStage, typing: str, fpath: str):
	with open(fpath, 'w+') as file:
		if stage == EGStage.LOCAL:
			file.write('types: {}, num preds: {}\n'.format(typing, len(new_edges)))
		else:
			file.write('{}  type propagation num preds: {}\n'.format(typing, len(new_edges)))

		for pred, edges in new_edges.items():
			file.write('predicate: {}\n'.format(pred))
			num_edges = sum(len(es) for edge_type, es in edges.items())
			file.write('num neighbors: {}\n'.format(num_edges))

			file.write('\n')
			if stage == EGStage.LOCAL:
				file.write('BInc sims\n')
			else:
				file.write('global sims\n')

			for key in [KEY_CONS_SUCC, KEY_PRECOND, KEY_CONS_FAIL]:
				file.write('%{}\n'.format(key))
				for edge in edges[key]:
					file.write('{} {:.4f}\n'.format(edge.pred, edge.score))
				file.write('\n')

			file.write('\n\n')


def filter_causal(graphs: EGraphCache):
	global ARGS

	if not os.path.exists(ARGS.outdir):
		os.makedirs(ARGS.outdir)

	seen_graphs = set()
	for typing, graph in graphs.items():
		if graph in seen_graphs:
			continue

		seen_graphs.add(graph)
		new_edges = defaultdict(dict)

		# Analyze graph
		for neg_pred in graph.nodes:
			if not neg_pred.startswith('NEG__'):
				continue

			pos_pred = neg_pred[len('NEG__'):]
			if pos_pred not in graph.nodes:
				continue

			pos_ents = graph.get_entailments(pos_pred)
			neg_ents = graph.get_entailments(neg_pred)

			try:
				cons_succ, precond, cons_fail = compare_entailments(pos_ents, neg_ents)
			except:
				print('ERROR: {} // {}'.format(typing, neg_pred))
				exit(1)

			if any([cons_succ, precond, cons_fail]):
				new_edges[pos_pred][KEY_CONS_SUCC] = cons_succ
				new_edges[pos_pred][KEY_PRECOND] = precond
				new_edges[pos_pred][KEY_CONS_FAIL] = cons_fail

		if new_edges:
			fname = typing
			file_suffix = '_local.txt' if graph.stage == EGStage.LOCAL else '_global.txt'
			fpath = os.path.join(ARGS.outdir, fname + file_suffix)
			if not ARGS.no_write:
				write_causal_graph(new_edges, graph.stage, typing, fpath)


def main():
	global ARGS, MARGIN
	ARGS = parser.parse_args()

	print('Reading graphs from {} ...'.format(ARGS.graphs))
	if ARGS.text_graphs:
		print('NYI; assuming local graphs')
		egcache = read_graphs(ARGS.graphs, EGStage.LOCAL, keep_forward=True)
	else:
		egcache = read_precomputed_EGs(ARGS.graphs)
	# fun = FUN_MAP[ARGS.fun]
	# print('Analyzing graph using {} comparison function...'.format(ARGS.fun))
	# analyze_causal(egcache, compare_fun=fun)
	print('Filtering graphs and classifying edges...')
	if ARGS.decision == 'margin':
		MARGIN = float(ARGS.margin)
		assert 0 <= MARGIN <= 1
		print('* Decision margin set to: {}'.format(MARGIN))

	filter_causal(egcache)
	print('Done')


parser = argparse.ArgumentParser(description='Read in and analyze entailment graphs for metarelations')
parser.add_argument('graphs', help='Path to file used for partition into Question set and Answer set')
parser.add_argument('outdir', help='location to place output graphs')
# parser.add_argument('--fun', default='pos', choices=['pos', 'inter', 'neg'], help='Comparison function between positive and negative entailment sets: keep only the selected subset')
parser.add_argument('--decision', default='strict', choices=['strict', 'gt', 'margin'], help='Comparison behavior')
parser.add_argument('--margin', default=MARGIN, help='Value for use with the margin decision function. Ranges from [0,1.0]')
parser.add_argument('--text-graphs', action='store_true', help='Flag if graphs are to be read in from raw text files')
parser.add_argument('--no-write', action='store_true', help='Flag if graphs are to be read in from raw text files')

if __name__ == '__main__':
	main()