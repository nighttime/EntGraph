from proposition import *
from collections import Counter
from typing import *

class Article:
	def __init__(self, art_ID, date):
		self.art_ID = art_ID
		self.date = date
		self.unary_props = []
		self.binary_props = []

	def add_unary(self, unary):
		self.unary_props.append(unary)

	def add_binary(self, binary):
		self.binary_props.append(binary)

	def remove_qa_pair(self, q, a):
		for i,u in enumerate(self.unary_props):
			if u.pred_desc() == q and u.args[0] == a:
				self.unary_props.pop(i)
				break

	def arg_mentions(self, arg: str) -> int:
		return len([p for p in self.unary_props + self.binary_props if arg in p.args])

	def feature_counts(self) -> Counter[str]:
		return Counter(self.named_entity_mentions())

	def named_entity_mentions(self, typed=True) -> List[str]:
		ents = []
		for p in self.unary_props + self.binary_props:
			for i, e_type in enumerate(p.entity_types):
				if e_type == 'E':
					ents.append(p.arg_desc()[i] if typed else p.args[i])
		return ents

	def remove_uprops(self, props):
		for p in props:
			self.unary_props.remove(p)


def read_source_data(source_fname: str) -> Tuple[List[Article], List[Prop], List[Prop]]:
	articles = {}
	unary_props = []
	binary_props = []
	with open(source_fname) as source_data:
		for line in source_data:
			l = json.loads(line)
			art_ID = int(l['articleId'])
			if art_ID not in articles:
				date = datetime.strptime(l['date'], '%b %d, %Y %X %p')
				articles[art_ID] = Article(art_ID, date)

			unaries_raw = [u['r'][1:-1] for u in l['rels_unary']]
			for i, u in enumerate(unaries_raw):
				parts = u.split('::')

				norm_pred, _ = normalize_predicate(parts[0])
				norm_ent = normalize_entity(parts[1])
				entity_type = parts[2]
				typing = get_type(norm_ent, 'E' in entity_type)

				if '\'' in norm_pred:
					continue

				if 'be.' in norm_pred:
					continue
				# norm_pred = norm_pred.replace('be.','')

				prop = Prop(norm_pred, [norm_ent])
				prop.set_entity_types(entity_type)
				prop.set_types([typing])

				unary_props.append(prop)
				articles[art_ID].add_unary(prop)

			binaries_raw = [u['r'][1:-1] for u in l['rels']]
			for i, b in enumerate(binaries_raw):
				parts = b.split('::')

				norm_pred, reversed_pred = normalize_predicate(parts[0])
				norm_ents = [normalize_entity(e) for e in parts[1:3]]
				entity_types = parts[3]
				typing = [get_type(norm_ents[i], 'E' in entity_types[i]) for i in range(2)]

				if reversed_pred:
					norm_ents = list(reversed(norm_ents))
					entity_types = entity_types[::-1]
					typing = list(reversed(typing))

				type_symmetric = typing[0] == typing[1]
				if type_symmetric:
					typing = [typing[0] + '_1', typing[1] + '_2']

				prop = Prop(norm_pred, norm_ents)
				prop.set_entity_types(entity_types)
				prop.set_types(typing)
				binary_props.append(prop)
				articles[art_ID].add_binary(prop)

				if type_symmetric:
					norm_ents = list(reversed(norm_ents))
					entity_types = entity_types[::-1]
					typing = list(reversed(typing))
					prop_rev = Prop(norm_pred, norm_ents)
					prop_rev.set_entity_types(entity_types)
					prop_rev.set_types(typing)
					binary_props.append(prop_rev)
					articles[art_ID].add_binary(prop_rev)

	arts = [a for artID, a in articles.items()]

	return arts, unary_props, binary_props