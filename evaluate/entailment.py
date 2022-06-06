from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import os
import sys
import pickle
import pdb
import numpy as np
import multiprocessing as mp

import utils
from proposition import Prop
from typing import *


class EGSpace(Enum):
	ONE_TYPE = 1
	TWO_TYPE = 2

class EGStage(Enum):
	LOCAL = 1
	GLOBAL = 2

class EdgeType(Enum):
	PRECONDITION = 'preconditions'
	CONSEQUENCE_SUCCESS = 'consequences_success'
	CONSEQUENCE_FAILURE = 'consequences_failure'

class EGMetric(Enum):
	BINC = 'BInc sims'
	WEEDS_PMI = 'Weed\'s PMI sim'
	WEEDS_PMI_PREC = 'Weed\'s PMI Precision sim'

class Entailment:
	# __slots__ = ['pred', 'basic_pred', 'score', 'direction', 'edge_type']
	def __init__(self, entailedPred: str, score: float, edge_type: Optional[EdgeType] = None):
		self.pred = entailedPred
		self.basic_pred = '#'.join(x.replace('_1', '').replace('_2', '') for x in self.pred.split('#'))
		self.score = score
		self.direction = 'forward'
		self.edge_type = edge_type

	def __str__(self):
		return '( {:.3f} => entailment: {} )'.format(self.score, self.pred)

	def __repr__(self):
		return str(self)

class BackEntailment(Entailment):
	# __slots__ = []
	def __init__(self, antecedentPred: str, score: float, edge_type: Optional[EdgeType] = None):
		super().__init__(antecedentPred, score, edge_type)
		self.direction = 'backward'

	def __str__(self):
		return '( antecedent: {} => {:.3f} )'.format(self.pred, self.score)

	def __repr__(self):
		return str(self)

class EntailmentGraph:
	typing = None
	reverse_typing = None
	space = None
	stage = None
	typed_edges = False
	metric = None
	nodes = None
	backmap = None
	edges = None
	edge_counts = None
	
	def __init__(self, fname: Optional[str] = None, keep_forward: bool = False, metric=EGMetric.BINC):
		if fname == None:
			return
		self.metric = metric
		nodes, edges = self._read_graph_from_file(fname)
		self._configure_graph(nodes, edges, keep_forward)

	def get_entailments(self, pred: str, edge_type: Optional[EdgeType] = None) -> List[Entailment]:
		assert self.edges
		if pred not in self.nodes:
			return []
		else:
			if edge_type:
				return [p for p in self.edges[pred] if p.edge_type == edge_type]
			else:
				return self.edges[pred]

	def get_entailments_count(self, pred: str) -> int:
		return self.edge_counts[pred] if pred in self.edge_counts else 0

	def get_antecedents(self, pred: str, edge_type: Optional[EdgeType] = None) -> Set[BackEntailment]:
		if pred not in self.nodes:
			return set()
		else:
			if edge_type:
				return {p for p in self.backmap[pred] if p.edge_type == edge_type}
			else:
				return self.backmap[pred]

	def get_antecedents_bare_pred(self, barepred: str) -> Set[BackEntailment]:
		forward = self.get_antecedents(barepred + '#' + self.typing)
		backward = self.get_antecedents(barepred + '#' + self.reverse_typing)
		return forward | backward

	def _configure_graph(self, nodes: Set[str], edges: Dict[str, List[Entailment]], keep_forward: bool):
		backmap = self._backmap_antecedents(edges)
		self.nodes = nodes
		self.backmap = backmap
		self.edges = edges if keep_forward else None
		self.edge_counts = {p:len(es) for p,es in edges.items()}

	def _configure_metadata(self, typing: str, space: EGSpace, stage: EGStage):
		self.typing = typing
		self.reverse_typing = '#'.join(list(reversed(typing.split('#'))))
		self.space = space
		self.stage = stage

	def _backmap_antecedents(self, edges: Dict[str, List[Entailment]]) -> Dict[str, Set[BackEntailment]]:
		backmap = defaultdict(set)
		for k,vlist in edges.items():
			for v in vlist:
				backmap[v.pred].add(BackEntailment(k, v.score, v.edge_type))
		return backmap

	@classmethod
	def from_edges(cls, edges: Dict[str, List[Entailment]], typing: str, space: EGSpace, stage: EGStage, keep_forward: bool = False) -> EntailmentGraph:
		new_graph = EntailmentGraph()
		new_graph._configure_metadata(typing, space, stage)
		nodes = set(edges.keys())
		new_graph._configure_graph(nodes, edges, keep_forward)
		return new_graph

	@classmethod
	def _strip_tag(cls, pred):
		tag_idx = pred.find(']')
		if tag_idx != -1:
			pred = pred[tag_idx+1:]
		return pred

	@classmethod
	def _unary_strip_second_type(cls, pred):
		if ',' not in pred:
			pred_parts = pred.split('#')
			return '#'.join(pred_parts[:2])
		return pred

	@classmethod
	def normalize_pred(cls, pred: str):
		clean_pred = EntailmentGraph._strip_tag(pred)
		clean_pred = EntailmentGraph._unary_strip_second_type(clean_pred)
		return clean_pred


	# _read_graph_from_file [UNLABELED EDGES ONLY]
	# def _read_graph_from_file(self, fname: str) -> Tuple[Set[str], Dict[str, List[Entailment]]]:
	# 	nodes = set()
	# 	edges = defaultdict(list)
	#
	# 	with open(fname) as file:
	# 		header = file.readline()
	# 		if not len(header):
	# 			return nodes, edges
	#
	# 		typing = ''
	#
	# 		if header.startswith('types:'):
	# 			self.stage = EGStage.LOCAL
	# 			typing = header[header.index(':') + 2:header.index(',')]
	# 		elif 'type propagation' in header:
	# 			self.stage = EGStage.GLOBAL
	# 			typing = header[:header.index(' ')]
	# 		else:
	# 			return nodes, edges
	#
	# 		if typing.count('#') == 0 or '#unary' in typing:
	# 			self.typing = typing.split('#')[0]
	# 			self.space = EGSpace.ONE_TYPE
	# 		else:
	# 			self.typing = typing
	# 			self.space = EGSpace.TWO_TYPE
	#
	# 		self.reverse_typing = '#'.join(reversed(self.typing.split('#')))
	#
	# 		current_pred = ''
	# 		line = next(file, None)
	# 		while line is not None:
	# 			line = line.strip()
	#
	# 			if line.startswith('predicate:'):
	# 				pred = line[line.index(':')+2:]
	# 				current_pred = EntailmentGraph.normalize_pred(pred)
	# 				self.nodes.add(current_pred)
	#
	# 			if any(t in line for t in ['BInc sims', 'iter 1 sims', 'global sims']):
	# 				line = next(file, None)
	# 				while line is not None:
	# 					line = line.strip()
	# 					if line == '':
	# 						break
	#
	# 					pred, score_text = line.split()
	# 					score = float(score_text)
	# 					if score < 0.01:
	# 						continue
	#
	# 					pred = EntailmentGraph.normalize_pred(pred)
	#
	# 					nodes.add(pred)
	# 					edges[current_pred].append(Entailment(pred, score))
	#
	# 					line = next(file, None)
	#
	# 			if line is None:
	# 				break
	#
	# 			line = next(file, None)
	#
	# 	return nodes, edges

	def _read_graph_from_file(self, fname: str) -> Tuple[Set[str], Dict[str, List[Entailment]]]:
		nodes = set()
		edges = defaultdict(list)

		with open(fname) as file:
			header = file.readline()
			if not len(header):
				return nodes, edges

			typing = ''

			if header.startswith('types:'):
				self.stage = EGStage.LOCAL
				typing = header[header.index(':') + 2:header.index(',')]
			elif 'type propagation' in header:
				self.stage = EGStage.GLOBAL
				typing = header[:header.index(' ')]
			else:
				return nodes, edges

			if typing.count('#') == 0 or '#unary' in typing:
				self.typing = typing.split('#')[0]
				self.space = EGSpace.ONE_TYPE
			else:
				self.typing = typing
				self.space = EGSpace.TWO_TYPE

			self.reverse_typing = '#'.join(reversed(self.typing.split('#')))

			current_pred = ''
			current_edge_type = None
			reading_edges = False
			line = next(file, None)
			while line is not None:
				line = line.strip()

				if line == '' or (line.startswith('num ') and 'neighbors' in line):
					reading_edges = False

				elif line.startswith('predicate:'):
					pred = line[line.index(':')+2:]
					current_pred = EntailmentGraph.normalize_pred(pred)
					nodes.add(current_pred)

				elif line.startswith('%'):
					edge_type = EdgeType(line[1:].strip())
					current_edge_type = edge_type
					self.typed_edges = True
					reading_edges = True

				elif self.stage == EGStage.GLOBAL and any(t in line for t in ['iter 0 sims', 'local sims']):
					reading_edges = False

				# BInc sims, Weed's PMI sim
				elif any(t in line for t in [self.metric.value, 'iter 1 sims', 'global sims']):
					reading_edges = True

				else:
					if reading_edges:
						pred, score_text = line.split()
						score = float(score_text)
						# if score >= 0.01:
						pred = EntailmentGraph.normalize_pred(pred)
						nodes.add(pred)
						edges[current_pred].append(Entailment(pred, score, current_edge_type))

				line = next(file, None)

		return nodes, edges

	def write_to_file(self, fpath: str):
		with open(fpath, 'w+') as file:
			if self.stage == EGStage.LOCAL:
				file.write('types: {}, num preds: {}\n'.format(self.typing, len(self.nodes)))
			else:
				file.write('{}  type propagation num preds: {}\n'.format(self.typing, len(self.nodes)))

			for pred, edges in self.edges.items():
				file.write('predicate: {}\n'.format(pred))
				file.write('num neighbors: {}\n'.format(len(edges)))
				file.write('\n')
				if self.stage == EGStage.LOCAL:
					file.write('BInc sims\n')
				else:
					file.write('global sims\n')

				if self.typed_edges:
					for edge_type in EdgeType:
						file.write('% {}\n'.format(edge_type.value))
						for e in self.edges:
							if e.edge_type == edge_type:
								file.write('{} {:.4f}\n'.format(e.pred, e.score))
						file.write('\n')
				else:
					for e in self.edges:
						file.write('{} {:.4f}\n'.format(e.pred, e.score))

				file.write('\n\n')

			return file

# Type definitions
EGraphCache = Dict[str, EntailmentGraph]

def query_all_graphs_for_prop(claim: Prop, ent_graphs: EGraphCache) -> List[Set[BackEntailment]]:
	antecedent_list = []
	for graph in ent_graphs.values():
		antecedents = graph.get_antecedents_bare_pred(claim.pred)
		antecedent_list.append(antecedents)
	return antecedent_list

def prop_recognized_in_graphs(prop: Prop, graphs: EGraphCache) -> bool:
	return any(len(a) > 0 for a in query_all_graphs_for_prop(prop, graphs))

def graph_files_in_dir(graph_dir: str, stage: Optional[EGStage], zipped: bool) -> List[str]:
	(_, _, filenames) = next(os.walk(graph_dir))
	if zipped:
		return [os.path.join(graph_dir, fname) for fname in filenames if fname.endswith('.pkl')]
	else:
		stage_ids = ['sim', 'local'] if stage == EGStage.LOCAL else ['binc', 'gsim', 'global']
		return [os.path.join(graph_dir, fname) for fname in filenames if
					any(fname.endswith('_' + stage_id + '.txt') for stage_id in stage_ids)]

def read_graphs(graph_dir: str, stage: EGStage, keep_forward=False, metric=EGMetric.BINC) -> EGraphCache:
	graph_fpaths = graph_files_in_dir(graph_dir, stage, False)
	graphs = []
	for i,g in enumerate(graph_fpaths):
		graphs.append(EntailmentGraph(g, keep_forward=keep_forward, metric=metric))
		utils.print_progress((i+1)/len(graph_fpaths), g)
	print()
	return make_EGraphCache(graphs)

def make_EGraphCache(graphs: List[EntailmentGraph]) -> EGraphCache:
	graphs = [g for g in graphs if g.typing is not None]
	g_forward = {g.typing: g for g in graphs}
	g_reverse = {g.reverse_typing: g for g in graphs}
	return {**g_forward, **g_reverse}

# Read in U->U and/or B->U entailment graphs
def load_graphs(graph_dir, message, ARGS) -> Optional[EGraphCache]:
	if graph_dir:
		print(message, end=' ', flush=True)
		if ARGS.text_EGs:
			stage = EGStage.LOCAL if ARGS.local else EGStage.GLOBAL
			graphs = read_graphs(graph_dir, stage, keep_forward=True)
		else:
			graphs = read_precomputed_EGs(graph_dir)
		return graphs


def open_graph(fpath):
	with open(fpath, 'rb') as f:
		return pickle.load(f)

def load_graphs_parallel(graph_dir, message, ARGS) -> Optional[EGraphCache]:
	if graph_dir:
		print(message, end=' ', flush=True)
		if ARGS.text_EGs:
			stage = EGStage.LOCAL if ARGS.local else EGStage.GLOBAL
			return read_graphs(graph_dir, stage, keep_forward=True)
		else:
			graph_fpaths = graph_files_in_dir(graph_dir, None, True)
			pool = mp.Pool()
			graphs = pool.map(open_graph, graph_fpaths)
			return make_EGraphCache(graphs)


_UU_EG_fname = 'UU_EGs.pkl'
_BU_EG_fname = 'BU_EGs.pkl'

def save_EGs(egcache: EGraphCache, fname: str):
	with open(fname + '.pkl', 'wb+') as f:
		pickle.dump(egcache, f, pickle.HIGHEST_PROTOCOL)

def save_EGs_separately(graphs: EGraphCache, foldername: str):
	if not os.path.exists(foldername):
		os.makedirs(foldername)
	seen = set()
	for i, graph in enumerate(graphs.values()):
		if graph.typing in seen:
			continue
		seen.add(graph.typing)
		with open(os.path.join(foldername, graph.typing + '.pkl'), 'wb+') as f:
			pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

def read_precomputed_EGs(fname: str) -> EGraphCache:
	# with open(fname, 'rb') as f:
	# 	return pickle.load(f)
	import gc
	f = open(fname, 'rb')
	gc.disable()
	res = pickle.load(f)
	gc.enable()
	f.close()
	return res


@dataclass
class EmbeddingCache:
	cache: np.ndarray
	id_map: Dict[str, int]

def load_similarity_cache(folder: str, model: str) -> EmbeddingCache:
	folder_name = folder if folder.endswith('/') else folder + '/'
	cache_fname = folder_name + model + '-prop_embs.npy'
	map_fname = folder_name + model + '-prop_emb_idx.pkl'
	with open(map_fname, 'rb') as f:
		cache_map = pickle.load(f)
	cache = np.load(cache_fname)
	return EmbeddingCache(cache, cache_map)


def main():
	global ARGS
	ARGS = parser.parse_args()

	graph_stage = EGStage.GLOBAL if ARGS.stage == 'global' else EGStage.LOCAL
	# egspace = EGSpace.ONE_TYPE if ARGS.valence == 'uv' else EGSpace.TWO_TYPE
	graph_metric = {'BInc': EGMetric.BINC,
					'Weeds_PMI': EGMetric.WEEDS_PMI,
					'Weeds_PMI_Prec': EGMetric.WEEDS_PMI_PREC
					}[ARGS.metric]

	print('Reading graphs from', ARGS.graphs)
	if ARGS.precomputed_in:
		egcache = read_precomputed_EGs(ARGS.graphs)
	else:
		egcache = read_graphs(ARGS.graphs, stage=graph_stage, keep_forward=ARGS.keep_forward, metric=graph_metric)

	print('Writing cache to', ARGS.outdir)
	if ARGS.separate_out:
		save_EGs_separately(egcache, ARGS.outdir)
	else:
		save_EGs(egcache, ARGS.outdir)
	print('Done')

parser = argparse.ArgumentParser(description='Read in and cache an entailment graph for easy use later')
parser.add_argument('graphs', help='Folder of raw graph files')
parser.add_argument('outdir', help='location to place output graph cache')
# parser.add_argument('--valence', required=True, choices=['uv', 'bv'], help='Graph valence: bv (bivalent; entailments from binaries) or uv (univalent; entailments from unaries)')
parser.add_argument('--precomputed-in', action='store_true', help='Input graphs are pickled together')
parser.add_argument('--separate-out', action='store_true', help='Pickle graphs separately in output')
parser.add_argument('--stage', default='local', choices=['local', 'global'], help='Graph stage: local or global')
parser.add_argument('--keep-forward', action='store_true', help='keep forward direction (otherwise only backward)')
parser.add_argument('--metric', default='BInc', choices=['BInc', 'Weeds_PMI', 'Weeds_PMI_Prec'], help='which metric to read in (usually from local graph files)')

if __name__ == '__main__':
	main()

