import os
import pickle
import pdb
from functools import lru_cache
from typing import *

# From (Java) entailment.Util
MODALS = ["can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought"]

class Prop:
	def __init__(self, predicate: str, args: List[str]):
		self.pred = predicate
		self.args = args
		self.types = []
		self.basic_types = []
		self.entity_types = ''

	@classmethod
	def from_descriptions(cls, pred_desc, arg_desc):
		pred_chunks = pred_desc.split('#')
		pred, types = pred_chunks[0], pred_chunks[1:]
		args = [a.split('#')[0] for a in arg_desc]
		prop = Prop(pred, args)
		prop.set_types(types)
		prop.entity_types = None
		return prop

	def set_args(self, args: List[str]):
		self.args = args

	def set_types(self, types: List[str]):
		self.types = types
		self.basic_types = [t.split('_')[0] for t in self.types]

	def set_entity_types(self, entity_types: str):
		self.entity_types = entity_types

	def arg_desc(self, numbered=False) -> List[str]:
		assert len(self.types) == len(self.args)
		ts = self.types if numbered else self.basic_types
		return [self.args[i] + '#' + ts[i] for i in range(len(self.args))]

	def pred_desc(self, reverse:bool=False) -> str:
		ts = reversed(self.types) if reverse else self.types
		return self.pred + ''.join('#' + t for t in ts)

	def type_desc(self, reverse:bool=False) -> str:
		ts = reversed(self.types) if reverse else self.types
		return '#'.join(ts).replace('_1', '').replace('_2', '')

	def prop_desc(self, reverse:bool=False) -> str:
		ts = reversed(self.types) if reverse else self.types
		args = reversed(self.args) if reverse else self.args
		return self.pred + ''.join('#' + t for t in ts) + str(args)

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__
		return False

	def __str__(self):
		return self.pred_desc() + ':' + str(self.arg_desc())

	def __repr__(self):
		return str(self)



def entity_is_only_NE(ent: str) -> bool:
	if len(ent) < 2:
		return False

	if ent[1:].lower() != ent[1:]:
		return True

	if any(c.upper() != c for c in ent):
		return False

	return True

def normalize_entity(ent: str) -> str:
	return ent.replace('-', ' ').replace('_', ' ').lower().strip()


def normalize_pred_part(pp: str) -> str:
	return '.'.join(filter(lambda x: x not in MODALS, pp.split('.')))


@lru_cache(maxsize=None)
def normalize_predicate(pred: str) -> Tuple[str, bool]:
	modifier = ''
	modifier_idx = pred.find('__')
	if modifier_idx != -1:
		modifier = pred[:modifier_idx+2]
		pred = pred[modifier_idx+2:]

	if not pred.startswith('(') or not pred.endswith(')') or ',' not in pred:
		return (pred, False)

	parts = pred[1:-1].lower().split(',')
	parts = [normalize_pred_part(pp) for pp in parts]

	p0 = parts[0]
	parts.sort()
	reversed_args = parts[0] != p0

	return (modifier + '(' + ','.join(parts) + ')', reversed_args)




def normalize_type(typing):
	if typing.startswith('/'):
		typing = typing[1:]
	subtyping_idx = typing.find('/')
	if subtyping_idx != -1:
		typing = typing[:subtyping_idx]
	if typing == '':
		typing = 'thing'
	return typing.strip()


_entity2type = None
_only_NE_entities = None

def load_entity_types(fname):
	global _entity2type, _only_NE_entities
	_entity2type = {}
	_only_NE_entities = set()

	with open(fname) as typelist:
		for line in typelist:
			parts = line.split('\t')

			ent, typing = parts[1:3]

			# Clean the entity
			only_NE = entity_is_only_NE(ent)
			e = normalize_entity(ent)

			# Clean the typing
			typing = normalize_type(typing)

			# Reject unnecessary additions
			if e in _entity2type and typing == 'thing':
				continue

			if e in _entity2type and only_NE:
				continue

			# Update cache
			if only_NE:
				_only_NE_entities.add(e)

			_entity2type[e] = typing


_e2t_fname = 'entity2type.pkl'
_onlyNE_fname = 'only_NE_entities.pkl'

def save_precomputed_entity_types(dirname):
	dirname = dirname if dirname.endswith('/') else dirname + '/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	global _entity2type, _only_NE_entities
	with open(dirname + _e2t_fname, 'wb+') as f:
		pickle.dump(_entity2type, f, pickle.HIGHEST_PROTOCOL)
	with open(dirname + _onlyNE_fname, 'wb+') as f:
		pickle.dump(_only_NE_entities, f, pickle.HIGHEST_PROTOCOL)

def load_precomputed_entity_types(dirname):
	dirname = dirname if dirname.endswith('/') else dirname + '/'
	global _entity2type, _only_NE_entities
	with open(dirname + _e2t_fname, 'rb') as f:
		_entity2type = pickle.load(f)
	with open(dirname + _onlyNE_fname, 'rb') as f:
		_only_NE_entities = pickle.load(f)

def get_type(ent, is_entity, typed=True):
	if not typed:
		return 'thing'

	global _entity2type, _only_NE_entities
	assert _entity2type is not None
	assert _only_NE_entities is not None
	
	ent_type = _entity2type.get(ent, 'thing')
	
	if not is_entity and ent in _only_NE_entities:
		ent_type = 'thing'

	return ent_type



