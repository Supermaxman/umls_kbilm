from abc import ABC, abstractmethod
import random
from typing import List, Set
import numpy as np
import math
import torch
import pytorch_lightning as pl
import torch.distributed as dist
import os

from umls_reader import read_umls
from umls import UmlsAtom, UmlsRelation

from kb_utils import RelationType, Concept, Relation, RelationExampleCreator


class NegativeRelationSampler(pl.Callback):
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
				num_samples += 1
				yield neg_rel_subj
				if num_samples >= self.negative_sample_size:
					break
				neg_rel_obj = Relation(subj=pos_relation.subj, rel_type=pos_relation.rel_type, obj=other_rel.obj)
				num_samples += 1
				yield neg_rel_obj
				if num_samples >= self.negative_sample_size:
					break


class UniformNegativeSampler(NegativeRelationSampler):
	def __init__(self, concepts: List[Concept], relations: Set[Relation], negative_sample_size: int, seed=0,
							 train_callback=False, val_callback=False, test_callback=False):
		super().__init__()
		self.negative_sample_size = negative_sample_size
		self.concepts = np.array(concepts)
		self.relations = relations
		self.seed = seed
		self.epoch = 0
		self.train_callback = train_callback
		self.val_callback = val_callback
		self.test_callback = test_callback
		self.gen = None

	def sample(self, pos_relation, batch_relations):
		num_samples = 0
		while num_samples < self.negative_sample_size:
			sample_idx = torch.randint(low=0, high=len(self.concepts), size=(1,), generator=self.gen)[0].item()
			sample_concept = self.concepts[sample_idx]
			replace_subject = torch.randint(low=0, high=2, size=(1,), generator=self.gen).bool()[0].item()
			if replace_subject:
				neg_rel = Relation(subj=sample_concept, rel_type=pos_relation.rel_type, obj=pos_relation.obj)
			else:
				neg_rel = Relation(subj=pos_relation.subj, rel_type=pos_relation.rel_type, obj=sample_concept)
			if neg_rel not in self.relations:
				num_samples += 1
				yield neg_rel

	def update_epoch(self, epoch):
		self.epoch = epoch
		try:
			rank = dist.get_rank()
		except AssertionError:
			if 'XRT_SHARD_ORDINAL' in os.environ:
				rank = int(os.environ['XRT_SHARD_ORDINAL'])
			else:
				rank = 0
				print('No process group initialized, using default seed...')

		self.gen = torch.Generator()
		self.gen.manual_seed(hash((self.seed, self.epoch, rank)))

	def on_train_epoch_start(self, trainer: pl.Trainer, pl_module):
		if self.train_callback:
			self.update_epoch(trainer.current_epoch)

	def on_validation_epoch_start(self, trainer, pl_module):
		if self.val_callback:
			self.update_epoch(epoch=0)

	def on_test_epoch_start(self, trainer, pl_module):
		if self.test_callback:
			self.update_epoch(epoch=0)
