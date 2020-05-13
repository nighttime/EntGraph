from collections import defaultdict
from enum import Enum
import os
import sys
import pickle
import pdb
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
		return '( {:.3f} {} => )'.format(self.score, self.pred)

	def __repr__(self):
		return str(self)

class BackEntailment(Entailment):
	def __init__(self, antecedentPred, score):
		super().__init__(antecedentPred, score)
		self.direction = 'backward'

	def __str__(self):
		return '( {:.3f} => {} )'.format(self.score, self.pred)

class EntailmentGraph:
	typing = None
	reverse_typing = None
	space = None
	stage = None
	nodes = set()
	backmap = defaultdict(set)
	# edges = defaultdict(list)
	
	def __init__(self, fname):
		nodes, edges = self.read_graph_from_file(fname)
		backmap = self.backmap_antecedents(edges)

		self.nodes = nodes
		self.backmap = backmap

	# def get_entailments(self, pred):
	# 	if pred not in self.nodes:
	# 		return []
	# 	else:
	# 		return self.edges[pred]

	def get_antecedents(self, pred):
		if not pred in self.nodes:
			return set()
		else:
			return self.backmap[pred]

	def backmap_antecedents(self, edges):
		backmap = defaultdict(set)
		for k,vlist in edges.items():
			for v in vlist:
				backmap[v.pred].add(BackEntailment(k, v.score))
		return backmap

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

	def normalize_pred(self, pred):
		pred = EntailmentGraph._strip_tag(pred)
		pred = EntailmentGraph._unary_strip_second_type(pred)
		return pred


	def read_graph_from_file(self, fname):
		nodes = set()
		edges = defaultdict(list)

		with open(fname) as file:
			header = file.readline()
			if not len(header):
				return

			typing = ''

			if header.startswith('types:'):
				self.stage = EGStage.LOCAL
				typing = header[header.index(':') + 2:header.index(',')]
			elif 'type propagation' in header:
				self.stage = EGStage.GLOBAL
				typing = header[:header.index(' ')]
			else:
				return

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
					current_pred = self.normalize_pred(pred)
					self.nodes.add(current_pred)

				if 'BInc sims' in line or 'iter 1 sims' in line:
					line = next(file, None)
					while line is not None:
						line = line.strip()
						if line == '':
							break

						pred, score_text = line.split()
						score = float(score_text)
						pred = self.normalize_pred(pred)

						nodes.add(pred)
						edges[current_pred].append(Entailment(pred, score))

						line = next(file, None)

				if line is None:
					break

				line = next(file, None)

		return nodes, edges

# Type definitions
EGraphCache = Dict[str, EntailmentGraph]

def read_graphs(graph_dir: str, ext: str='sim') -> EGraphCache:
	(_, _, filenames) = next(os.walk(graph_dir))
	graph_fpaths = [os.path.join(graph_dir, fname) for fname in filenames if fname.endswith('_' + ext + '.txt')]
	graphs = [EntailmentGraph(g) for g in graph_fpaths]
	g_forward = {g.typing: g for g in graphs}
	g_reverse = {g.reverse_typing: g for g in graphs}
	return {**g_forward, **g_reverse}


_UU_EG_fname = 'UU_EGs.pkl'
_BU_EG_fname = 'BU_EGs.pkl'

def save_EGs(egcache: EGraphCache, fname: str, egspace: EGSpace):
	with open(fname + '.pkl', 'wb+') as f:
		pickle.dump(egcache, f, pickle.HIGHEST_PROTOCOL)

def load_precomputed_EGs(fname: str) -> EGraphCache:
	with open(fname, 'rb') as f:
		return pickle.load(f)

if __name__ == '__main__':
	if len(sys.argv) != 5:
		print('Cache EGraphs: <-g|-l> <type space: 1 or 2> <graph dir> <output cache dir>')
	graph_ext = 'binc' if sys.argv[1] == '-g' else 'sim'
	egspace = {'1': EGSpace.ONE_TYPE, '2': EGSpace.TWO_TYPE}[sys.argv[2]]
	in_dir = sys.argv[3]
	out_dir = sys.argv[4]

	print('Reading graphs from', in_dir)
	egcache = read_graphs(in_dir, ext=graph_ext)
	print('Writing cache to', out_dir)
	save_EGs(egcache, out_dir, egspace)
