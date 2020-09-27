import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import random
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist

from umls_reader import read_umls
from umls import UmlsAtom, UmlsRelation
from kb_utils import RelationType, Concept, Relation, RelationExampleCreator
from sample_utils import NegativeRelationSampler


def load_rel_merge_mapping(filepath):
	rel_merge_mapping = {}
	with open(filepath, 'r') as f:
		for line in f:
			s, t = line.strip().split(',')
			rel_merge_mapping[s] = t
	return rel_merge_mapping


def load_rela_mapping(filepath):
	rela_mapping = {}
	with open(filepath, 'r') as f:
		for line in f:
			rela, rela_text = line.strip().split('\t')
			rela = rela.strip()
			rela_text = rela_text.strip()
			rela_mapping[rela] = rela_text
	return rela_mapping


def load_rel_mapping(filepath):
	rel_mapping = {}
	with open(filepath, 'r') as f:
		for line in f:
			rela, rel_text = line.strip().split('\t')
			rela = rela.strip()
			rel_text = rel_text.strip()
			rel_mapping[rela] = rel_text
	return rel_mapping


def split_data(data, train_ratio=0.7, dev_ratio=0.1):
	train_size = int(len(data) * train_ratio)
	dev_size = int(len(data) * dev_ratio)
	train_data = data[:train_size]
	dev_data = data[train_size:train_size + dev_size]
	test_data = data[train_size + dev_size:]

	return train_data, dev_data, test_data


def load_umls(umls_directory, data_folder='./data'):

	cache_file = os.path.join(data_folder, 'umls_cache.pickle')
	if os.path.exists(cache_file):
		logging.info('Loading cache file...')
		with open(cache_file, 'rb') as f:
			concepts, relation_types, relations = pickle.load(f)
		return concepts, relation_types, relations

	logging.info('No cache file found, loading from umls directory...')
	rrf_file = os.path.join(umls_directory, 'META', 'MRREL.RRF')
	conso_file = os.path.join(umls_directory, 'META', 'MRCONSO.RRF')
	# TODO cache return of function in pickle, no need to generate every time.
	rel_merge_mapping = load_rel_merge_mapping(os.path.join(data_folder, 'rel_merge_mapping.txt'))
	rel_mapping = load_rel_mapping(os.path.join(data_folder, 'rel_desc.txt'))
	rela_mapping = load_rela_mapping(os.path.join(data_folder, 'rela_desc.txt'))

	valid_rel_cuis = set(rel_merge_mapping.keys())
	languages = {'ENG'}
	logging.info(f'Reading umls concepts...')
	triples = set()

	def is_preferred(x):
		return x.ts == 'P' and x.stt == 'PF' and x.ispref == 'Y'

	def umls_concept_filter(x):
		# filter out non-english atoms
		if x.lat not in languages:
			return False
		# Ignore non-preferred atoms
		if is_preferred(x):
			return True
		return False

	concept_iter = read_umls(
		conso_file,
		UmlsAtom,
		umls_filter=umls_concept_filter
	)
	seen_cuis = set()
	total_matching_concept_count = 3285966
	# First pass through to get all possible cuis which we have atoms for.
	for atom in tqdm(concept_iter, desc="reading", total=total_matching_concept_count):
		seen_cuis.add(atom.cui)
	logging.info(f'Matching cui count: {len(seen_cuis)}')

	def umls_rel_filter(x):
		# remove recursive relations
		if x.cui2 == x.cui1:
			return False
		# ignore siblings, CHD is enough to infer
		if x.rel == 'SIB':
			return False
		# ignore PAR, CHD is reflexive
		if x.rel == 'PAR':
			return False
		# ignore RO with no relA, not descriptive
		if x.rel == 'RO' and x.rela == '':
			return False
		# reflexive with AQ
		if x.rel == 'QB':
			return False
		# too vague
		if x.rel == 'RB':
			return False
		# removes rels which have too few instances to keep around
		if f'{x.rel}:{x.rela}' not in valid_rel_cuis:
			return False
		# removes rels which do not have matching atoms/cuis
		if x.cui1 not in seen_cuis or x.cui2 not in seen_cuis:
			return False
		return True

	rel_iter = read_umls(
		rrf_file,
		UmlsRelation,
		umls_filter=umls_rel_filter
	)
	rel_count = 0
	total_matching_rel_count = 12833112
	relation_types = {}
	seen_rel_concept_cuis = set()
	# now get all rels which match our requirements and also have atoms.
	for rel in tqdm(rel_iter, desc="reading", total=total_matching_rel_count):
		rel_cui = rel_merge_mapping[f'{rel.rel}:{rel.rela}']
		if rel_cui not in relation_types:
			cui_rel, cui_rela = rel_cui.split(':')
			# if there is no rela then we use rel text
			if cui_rela == '':
				rel_text = rel_mapping[cui_rel]
			else:
				if cui_rela not in rela_mapping:
					rela_mapping[cui_rela] = ' '.join(cui_rela.split('_'))
					logging.info(f'rela {cui_rela} not found in text mapping, defaulting to {rela_mapping[cui_rela]}.')
				rel_text = rela_mapping[cui_rela]
			relation_types[rel_cui] = RelationType(rel_cui, rel_text)
		# TODO double check order here
		seen_rel_concept_cuis.add(rel.cui1)
		seen_rel_concept_cuis.add(rel.cui2)
		triples.add((rel.cui2, rel_cui, rel.cui1))
		rel_count += 1
	logging.info(f'Matching rel count: {rel_count}')
	logging.info(f'Rel types: {len(relation_types)}')

	def umls_atom_filter(x):
		# filter out non-english atoms
		# TODO allow other language atoms?
		if x.lat not in languages:
			return False
		# ignore atoms for concepts of which there are no relations.
		if x.cui not in seen_rel_concept_cuis:
			return False
		return True

	logging.info(f'Reading umls atoms...')
	atom_iter = read_umls(
		conso_file,
		UmlsAtom,
		umls_filter=umls_atom_filter
	)
	atom_count = 0
	# total_matching_atom_count = 6873557
	total_matching_atom_count = 7753235
	concepts = {}

	# finally, get atoms for only concepts which we have relations for.
	# TODO expand to non-primary atoms
	for atom in tqdm(atom_iter, desc="reading", total=total_matching_atom_count):
		if atom.cui not in concepts and is_preferred(atom):
			concepts[atom.cui] = Concept(atom.cui, atom.string)
			atom_count += 1

	logging.info(f'Read {len(concepts)} concepts')
	relations = []
	for subj_cui, rel_cui, obj_cui in tqdm(triples, desc='creating relations', total=len(triples)):
		relation = Relation(
			subj=concepts[subj_cui],
			rel_type=relation_types[rel_cui],
			obj=concepts[obj_cui]
		)
		relations.append(relation)

	random.shuffle(relations)
	logging.info('Saving cache file...')
	with open(cache_file, 'wb') as f:
		pickle.dump((concepts, relation_types, relations), f)
	return concepts, relation_types, relations


def get_optimizer_params(model, weight_decay):
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_params = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
		 'weight_decay': weight_decay},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

	return optimizer_params


class UmlsRelationDataset(Dataset):
	def __init__(self, relations, concepts, example_creator, tokenizer, sampler, negative_sample_size, max_seq_len):
		self.relations = relations
		self.concepts = concepts
		self.example_creator = example_creator
		self.tokenizer = tokenizer
		self.sampler = sampler
		self.negative_sample_size = negative_sample_size
		self.max_seq_len = max_seq_len
		self.relations_set = set(relations)

	def __len__(self):
		return len(self.relations)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		examples = []
		pos_relation = self.relations[idx]

		examples.append(self.example_creator.create(pos_relation))

		negative_examples = []
		while len(negative_examples) < self.negative_sample_size:
			neg_relation = self.sampler.sample(pos_relation, self.concepts)
			if neg_relation not in self.relations_set:
				negative_examples.append(self.example_creator(neg_relation))

		examples.extend(negative_examples)
		tokenizer_batch = self.tokenizer.batch_encode_plus(
			batch_text_or_text_pairs=examples,
			add_special_tokens=True,
			padding='max_length',
			return_tensors='pt',
			truncation=True,
			max_length=self.max_seq_len
		)
		example = {
			'input_ids': tokenizer_batch['input_ids'],
			'attention_mask': tokenizer_batch['attention_mask']
		}

		return example


class RelationCollator(object):
	def __init__(
			self, tokenizer, example_creator: RelationExampleCreator, neg_sampler: NegativeRelationSampler,
			max_seq_len: int, force_max_seq_len: bool):
		super().__init__()
		self.tokenizer = tokenizer
		self.example_creator = example_creator
		self.neg_sampler = neg_sampler
		self.max_seq_len = max_seq_len
		self.force_max_seq_len = force_max_seq_len

	def __call__(self, relations):
		# creates text examples
		batch_negative_sample_size = None
		examples = []
		for pos_rel in relations:
			pos_example = self.example_creator.create(pos_rel)
			examples.append(pos_example)
			num_samples = 0
			for neg_rel in self.neg_sampler.sample(pos_rel, relations):
				neg_example = self.example_creator.create(neg_rel)
				examples.append(neg_example)
				num_samples += 1
			if batch_negative_sample_size is None:
				batch_negative_sample_size = num_samples

		# "input_ids": batch["input_ids"].to(device),
		# "attention_mask": batch["attention_mask"].to(device),
		tokenizer_batch = self.tokenizer.batch_encode_plus(
			batch_text_or_text_pairs=examples,
			add_special_tokens=True,
			padding='max_length' if self.force_max_seq_len else 'longest',
			return_tensors='pt',
			truncation=True,
			max_length=self.max_seq_len
		)
		batch_size = len(relations)
		sample_size = batch_negative_sample_size + 1
		max_seq_len = tokenizer_batch['input_ids'].shape[1]
		batch = {
			'input_ids': tokenizer_batch['input_ids'].view(batch_size, sample_size, max_seq_len),
			'attention_mask': tokenizer_batch['attention_mask'].view(batch_size, sample_size, max_seq_len)
		}

		return batch
