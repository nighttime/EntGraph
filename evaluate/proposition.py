import argparse
import os
import pickle
import json
from datetime import datetime
import pdb
from functools import lru_cache
import copy
# From (Java) entailment.Util
import reference
from typing import *

Prop_Type = TypeVar('Prop_Type', bound='Prop')

class Prop:
	def __init__(self, predicate: str, args: List[str], date: Optional[datetime] = None):
		self.pred = predicate
		self.args = args
		self.types = []
		self.basic_types = []
		self.entity_types = ''
		self.date = date

	@classmethod
	def from_descriptions(cls: type, pred_desc: str, arg_desc: List[str]) -> Prop_Type:
		pred_chunks = pred_desc.split('#')
		pred, types = pred_chunks[0], pred_chunks[1:]
		args = [a.split('#')[0] for a in arg_desc]
		prop = Prop(pred, args)
		prop.set_types(types)
		prop.entity_types = None
		return prop

	@classmethod
	def with_swapped_pred(cls: type, prop: Prop_Type, pred_swap: str) -> Prop_Type:
		new_prop = copy.deepcopy(prop)
		# new_prop.pred = pred_swap + prop.pred[prop.pred.find('.'):]
		new_prop.pred = Prop.swap_pred(new_prop.pred, pred_swap)
		return new_prop

	# @classmethod
	# def with_new_pred(cls: type, prop: Prop_Type, new_pred_desc: str) -> Prop_Type:
	# 	desc_parts = new_pred_desc.split('#')
	# 	new_pred, new_types = desc_parts[0], desc_parts[1:]
	# 	assert new_types == prop.types or new_types == list(reversed(prop.types))
	#
	# 	new_prop = copy.deepcopy(prop)
	# 	new_prop.pred = new_pred
	# 	if new_types == list(reversed(prop.types)):
	# 		new_prop.set_types(new_types)
	# 		new_prop.set_args(list(reversed(new_prop.args)))
	# 		if new_prop.entity_types:
	# 			new_prop.set_entity_types(str(reversed(new_prop.entity_types)))
	#
	# 	return new_prop

	@classmethod
	def with_new_pred(cls: type, prop: Prop_Type, new_pred_desc: str) -> Prop_Type:
		desc_parts = new_pred_desc.split('#')
		new_pred, new_types = desc_parts[0], desc_parts[1:]

		reverse_args = False

		if len(prop.types) > 1:
			assert prop.types[0] != prop.types[1] and new_types[0] != new_types[1]
			if (prop.types == list(reversed(new_types))) \
					or ('_1' in prop.types[0] and '_2' in new_types[0]) \
					or ('_2' not in prop.types[0] and '_2' in new_types[0]):
				reverse_args = True

		new_prop = copy.deepcopy(prop)
		new_prop.pred = new_pred
		new_prop.set_types(new_types)
		if reverse_args:
			new_prop.set_args(list(reversed(new_prop.args)))
			if new_prop.entity_types:
				new_prop.set_entity_types(str(reversed(new_prop.entity_types)))

		return new_prop


	# Input: predicate OR predicate description
	# Output: the same, but with the base word replaced with the new word
	@classmethod
	def swap_pred(cls, pred_desc: str, replacement_word: str) -> str:
		base_word = extract_predicate_base_term(pred_desc)
		return pred_desc.replace(base_word, replacement_word)
		# if pred_desc.startswith('('):
		# 	base_word = extract_predicate_base_term(pred_desc)
		# 	return pred_desc.replace(base_word, replacement_word)
		# else:
		# 	return replacement_word + pred_desc[pred_desc.find('.'):]

	def set_args(self, args: List[str]):
		self.args = args

	def set_types(self, types: List[str]):
		self.types = types
		self.basic_types = [t.replace('_1', '').replace('_2', '') for t in self.types]

	def set_entity_types(self, entity_types: str):
		self.entity_types = entity_types

	def set_date(self, date: datetime):
		self.date = date

	def arg_desc(self, numbered=False) -> List[str]:
		assert len(self.types) == len(self.args)
		ts = self.types if numbered else self.basic_types
		return [self.args[i] + '#' + ts[i] for i in range(len(self.args))]

	def pred_desc(self, reverse:bool=False) -> str:
		ts = reversed(self.types) if reverse else self.types
		return self.pred + ''.join('#' + t for t in ts)

	def type_desc(self, reverse:bool=False, keep_ids=False) -> str:
		ts = reversed(self.types) if reverse else self.types
		typing = '#'.join(ts)
		if keep_ids:
			return typing
		else:
			return typing.replace('_1', '').replace('_2', '')

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

	def __hash__(self):
		return hash(str(self))


def decompose_binary_pred(prop: Prop) -> Tuple[str, str]:
	assert len(prop.args) == 2
	binary_pred_halves = prop.pred[prop.pred.find('(') + 1:prop.pred.find(')')].split(',')
	return binary_pred_halves[0], binary_pred_halves[1]

def binary_pred_root(prop: Prop) -> str:
	assert len(prop.args) == 2
	left, right = decompose_binary_pred(prop)
	pred_parts_l, pred_parts_r = left.split('.'), right.split('.')

	i = 0
	while i < len(pred_parts_l) and i < len(pred_parts_r) and pred_parts_l[i] == pred_parts_r[i]:
		i += 1

	return '.'.join(pred_parts_l[:i])

def extract_predicate_base(pred: str) -> str:
	unary = '(' not in pred
	mod = ''
	if '__' in pred:
		mod, pred = pred.split('__')

	if unary:
		parts = pred.split('.')[:-1]
	else:
		parts = pred[pred.index(',')+1:pred.index(')')].split('.')[:-1]

	if unary and parts[0] == 'be':
		parts = parts[1:]

	# if len(parts) > 1 and parts[-1] in reference.PREPOSITIONS:
	# 	parts = parts[:-1]

	return ' '.join(parts).replace('-', ' ')

# DEPRECATED
def extract_predicate_base_term(pred: str) -> str:
	if all(c in pred for c in '(,)'):
		parts = pred.split(',')[0].split('.')
	else:
		parts = pred.split('.')
	if pred.startswith('be.') and len(parts) > 2:
		return '.'.join(parts[:2])
	root = parts[0]
	if root.startswith('('):
		root = root[1:]
	return root

def format_prop_ppdb_lookup(prop: Prop):
	formatted = extract_predicate_base(prop.pred)
	# if formatted.startswith('be.'):
	# 	formatted = formatted[3:]
	return formatted

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
	return '.'.join(filter(lambda x: x not in reference.MODAL_VERBS, pp.split('.')))


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


def load_majority_entity_types(fname):
	global _entity2type, _only_NE_entities
	_entity2type = {}
	_only_NE_entities = set()

	with open(fname) as typelist:
		for line in typelist:
			parts = line.split('\t')

			# ent, typing = parts[1:3]
			ent = parts[1]
			typings = parts[2].split()

			# Clean the entity
			only_NE = entity_is_only_NE(ent)
			e = normalize_entity(ent)

			# Clean the typings
			if typings:
				typing = Counter([normalize_type(t) for t in typings]).most_common()[0][0]
			else:
				typing = 'thing'

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


# _sub_pairs_fname = 'substitution_pairs.json'
# _sub_pairs_fname = 'neg_swap_person.json'
# _sub_pairs_fname = 'neg_swap_person_filtered.json'

def read_substitution_pairs(fname: str) -> Dict[str, Dict[str, List[str]]]:
	# dirname = dirname if dirname.endswith('/') else dirname + '/'
	# global _sub_pairs_fname
	with open(fname) as f:
		sub_dict = json.load(f)
		return sub_dict


def main():
	global ARGS
	ARGS = parser.parse_args()

	if ARGS.make_type_cache:
		print('Generating new type cache...')
		type_fpath = '../data/freebase_types/entity2Types.txt'
		load_majority_entity_types(type_fpath)
		save_precomputed_entity_types(ARGS.make_type_cache)
		print('Done')



parser = argparse.ArgumentParser(description='Proposition utilities')
parser.add_argument('--make-type-cache', help='Recache the entity type file')

if __name__ ==  '__main__':
	main()