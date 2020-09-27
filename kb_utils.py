from abc import ABC, abstractmethod


class Concept(object):
	def __init__(self, concept_id: str, concept_name: str):
		self.concept_id = concept_id
		self.concept_name = concept_name

	def __str__(self):
		return f'{self.concept_id}:{self.concept_name}'

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash(self.concept_id)

	def __eq__(self, other):
		if not isinstance(other, Concept):
			return False
		return self.concept_id == other.concept_id


class RelationType(object):
	def __init__(self, relation_id: str, relation_name: str):
		self.relation_id = relation_id
		self.relation_name = relation_name

	def __str__(self):
		return f'{self.relation_id}:{self.relation_name}'

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash(self.relation_id)

	def __eq__(self, other):
		if not isinstance(other, RelationType):
			return False
		return self.relation_id == other.relation_id


class Relation(object):
	def __init__(self, subj: Concept, rel_type: RelationType, obj: Concept):
		self.subj = subj
		self.rel_type = rel_type
		self.obj = obj

	def __str__(self):
		return f'{self.subj}\t{self.rel_type}\t{self.obj}'

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash((self.subj.concept_id, self.rel_type.relation_id, self.obj.concept_id))

	def __eq__(self, other):
		if not isinstance(other, Relation):
			return False
		return self.subj == other.subj and self.rel_type == other.rel_type and self.obj == other.obj


class RelationExampleCreator(ABC):
	def __init__(self):
		pass

	@abstractmethod
	def create(self, rel: Relation):
		pass


class NameRelationExampleCreator(RelationExampleCreator):
	def __init__(self):
		super().__init__()

	def create(self, rel: Relation):
		return f'{rel.subj.concept_name} {rel.rel_type.relation_name.lower()} {rel.obj.concept_name}'
