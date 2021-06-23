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
from proposition import Prop
from typing import *


class EGSpace(Enum):
	ONE_TYPE = 1
	TWO_TYPE = 2

class EGStage(Enum):
	LOCAL = 1
	GLOBAL = 2

class Entailment:
	def __init__(self, entailedPred, score):
		self.pred = entailedPred
		self.basic_pred = '#'.join(x.replace('_1', '').replace('_2', '') for x in self.pred.split('#'))
		self.score = score
		self.direction = 'forward'

	def __str__(self):
		return '( {:.3f} => entailment: {} )'.format(self.score, self.pred)

	def __repr__(self):
		return str(self)

class BackEntailment(Entailment):
	def __init__(self, antecedentPred, score):
		super().__init__(antecedentPred, score)
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
	nodes = set()
	backmap = defaultdict(set)
	edges = defaultdict(list)
	
	def __init__(self, fname: Optional[str] = None, keep_forward: bool = False):
		if fname == None:
			return
		nodes, edges = self._read_graph_from_file(fname)
		self._configure_graph(nodes, edges, keep_forward)

	def get_entailments(self, pred):
		if pred not in self.nodes:
			return []
		else:
			return self.edges[pred]

	def get_antecedents(self, pred) -> Set[BackEntailment]:
		if pred not in self.nodes:
			return set()
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

	def _configure_metadata(self, typing: str, space: EGSpace, stage: EGStage):
		self.typing = typing
		self.reverse_typing = '#'.join(list(reversed(typing.split('#'))))
		self.space = space
		self.stage = stage

	def _backmap_antecedents(self, edges: Dict[str, List[Entailment]]) -> Dict[str, Set[BackEntailment]]:
		backmap = defaultdict(set)
		for k,vlist in edges.items():
			for v in vlist:
				backmap[v.pred].add(BackEntailment(k, v.score))
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
			line = next(file, None)
			while line is not None:
				line = line.strip()

				if line.startswith('predicate:'):
					pred = line[line.index(':')+2:]
					current_pred = EntailmentGraph.normalize_pred(pred)
					self.nodes.add(current_pred)

				if any(t in line for t in ['BInc sims', 'iter 1 sims', 'global sims']):
					line = next(file, None)
					while line is not None:
						line = line.strip()
						if line == '':
							break

						pred, score_text = line.split()
						score = float(score_text)
						if score < 0.01:
							continue

						pred = EntailmentGraph.normalize_pred(pred)

						nodes.add(pred)
						edges[current_pred].append(Entailment(pred, score))

						line = next(file, None)

				if line is None:
					break

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

				for edge in edges:
					file.write('{} {:.4f}\n'.format(edge.pred, edge.score))

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

def read_graphs(graph_dir: str, stage: EGStage, keep_forward=False) -> EGraphCache:
	(_, _, filenames) = next(os.walk(graph_dir))
	exts = ['sim'] if stage == EGStage.LOCAL else ['binc', 'gsim']
	graph_fpaths = [os.path.join(graph_dir, fname) for fname in filenames if any(fname.endswith('_' + ext + '.txt') for ext in exts)]
	graphs = list(filter(lambda x: x.typing is not None, [EntailmentGraph(g, keep_forward=keep_forward) for g in graph_fpaths]))
	g_forward = {g.typing: g for g in graphs}
	g_reverse = {g.reverse_typing: g for g in graphs}
	return {**g_forward, **g_reverse}


_UU_EG_fname = 'UU_EGs.pkl'
_BU_EG_fname = 'BU_EGs.pkl'

def save_EGs(egcache: EGraphCache, fname: str):
	with open(fname + '.pkl', 'wb+') as f:
		pickle.dump(egcache, f, pickle.HIGHEST_PROTOCOL)

def read_precomputed_EGs(fname: str) -> EGraphCache:
	with open(fname, 'rb') as f:
		return pickle.load(f)


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

	print('Reading graphs from', ARGS.graphs)
	egcache = read_graphs(ARGS.graphs, stage=graph_stage, keep_forward=ARGS.keep_forward)
	print('Writing cache to', ARGS.outdir)
	save_EGs(egcache, ARGS.outdir)

parser = argparse.ArgumentParser(description='Read in and cache an entailment graph for easy use later')
parser.add_argument('graphs', help='Folder of raw graph files')
parser.add_argument('outdir', help='location to place output graph cache')
# parser.add_argument('--valence', required=True, choices=['uv', 'bv'], help='Graph valence: bv (bivalent; entailments from binaries) or uv (univalent; entailments from unaries)')
parser.add_argument('--stage', required=True, choices=['local', 'global'], help='Graph stage: local or global')
parser.add_argument('--keep-forward', action='store_true', help='keep forward direction (otherwise only backward)')

if __name__ == '__main__':
	main()

