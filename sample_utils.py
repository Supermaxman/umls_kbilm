from abc import ABC, abstractmethod
import random
from typing import List
import numpy as np
import torch.distributed as dist

from umls_reader import read_umls
from umls import UmlsAtom, UmlsRelation

from kb_utils import RelationType, Concept, Relation, RelationExampleCreator


class NegativeRelationSampler(ABC):
	def __init__(self):
		pass

	@abstractmethod
	def sample(self, pos_relation, batch_relations):
		pass


class BatchNegativeSampler(NegativeRelationSampler):
	def __init__(self, negative_sample_size: int):
		super().__init__()
		self.negative_sample_size = negative_sample_size

	def sample(self, pos_relation, batch_relations):
		# creates text examples
		num_samples = 0
		for other_rel in batch_relations:
			if other_rel != pos_relation:
				neg_rel_subj = Relation(subj=other_rel.subj, rel_type=pos_relation.rel_type, obj=pos_relation.obj)
				neg_rel_obj = Relation(subj=pos_relation.subj, rel_type=pos_relation.rel_type, obj=other_rel.obj)
				if num_samples >= self.negative_sample_size:
					break
				num_samples += 1
				yield neg_rel_subj
				if num_samples >= self.negative_sample_size:
					break
				num_samples += 1
				yield neg_rel_obj


class UniformNegativeSampler(NegativeRelationSampler):
	def __init__(self, concepts: List[Concept], negative_sample_size: int):
		super().__init__()
		self.negative_sample_size = negative_sample_size
		self.concepts = np.array(concepts)
		try:
			self.rank = dist.get_rank()
			self.world_size = dist.get_world_size()
		except:
			self.rank = None
			self.world_size = None
		print(f'UniformNegativeSampler rank={self.rank}, world_size={self.world_size}')

	def sample(self, pos_relation, batch_relations):

		sample_idxs = np.random.randint(len(self.concepts), size=self.negative_sample_size)
		sample_concepts = self.concepts[sample_idxs]
		for sample_concept in sample_concepts:
			if random.random() < 0.5:
				neg_rel = Relation(subj=sample_concept, rel_type=pos_relation.rel_type, obj=pos_relation.obj)
			else:
				neg_rel = Relation(subj=pos_relation.subj, rel_type=pos_relation.rel_type, obj=sample_concept)
			yield neg_rel

