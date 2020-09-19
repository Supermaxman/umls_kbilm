import os
from . import umls_reader
from . import umls
from kb_utils import RelationType, Concept, Relation, RelationExampleCreator
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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


def load_umls(umls_directory, data_folder='./data'):
	rrf_file = os.path.join(umls_directory, 'META', 'MRREL.RRF')
	conso_file = os.path.join(umls_directory, 'META', 'MRCONSO.RRF')

	rel_merge_mapping = load_rel_merge_mapping(os.path.join(data_folder, 'rel_merge_mapping.txt'))
	rel_mapping = load_rel_mapping(os.path.join(data_folder, 'rel_desc.txt'))
	rela_mapping = load_rela_mapping(os.path.join(data_folder, 'rela_desc.txt'))

	valid_rel_cuis = set(rel_merge_mapping.keys())
	languages = {'ENG'}
	print(f'Reading umls concepts...')
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

	concept_iter = umls_reader.read_umls(
		conso_file,
		umls.UmlsAtom,
		umls_filter=umls_concept_filter
	)
	seen_cuis = set()
	total_matching_concept_count = 3285966
	# First pass through to get all possible cuis which we have atoms for.
	for atom in tqdm(concept_iter, desc="reading", total=total_matching_concept_count):
		seen_cuis.add(atom.cui)
	print(f'Matching cui count: {len(seen_cuis)}')

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

	rel_iter = umls_reader.read_umls(
		rrf_file,
		umls.UmlsRelation,
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
					print(f'rela {cui_rela} not found in text mapping, defaulting to {rela_mapping[cui_rela]}.')
				rel_text = rela_mapping[cui_rela]
			relation_types[rel_cui] = RelationType(rel_cui, rel_text)
		# TODO double check order here
		seen_rel_concept_cuis.add(rel.cui1)
		seen_rel_concept_cuis.add(rel.cui2)
		triples.add((rel.cui1, rel_cui, rel.cui2))
		rel_count += 1
	print(f'Matching rel count: {rel_count}')
	print(f'Rel types: {len(relation_types)}')

	def umls_atom_filter(x):
		# filter out non-english atoms
		# TODO allow other language atoms?
		if x.lat not in languages:
			return False
		# ignore atoms for concepts of which there are no relations.
		if x.cui not in seen_rel_concept_cuis:
			return False
		return True

	print(f'Reading umls atoms...')
	atom_iter = umls_reader.read_umls(
		conso_file,
		umls.UmlsAtom,
		umls_filter=umls_atom_filter
	)
	atom_count = 0
	total_matching_atom_count = 6873557
	concepts = {}

	# finally, get atoms for only concepts which we have relations for.
	# TODO expand to non-primary atoms
	for atom in tqdm(atom_iter, desc="reading", total=total_matching_atom_count):
		if atom.cui not in concepts and is_preferred(atom):
			concepts[atom.cui] = Concept(atom.cui, atom.string)
			atom_count += 1

	print(f'Read {len(concepts)} concepts')
	relations = []
	for subj_cui, rel_cui, obj_cui in tqdm(triples, desc='creating relations', total=len(triples)):
		relation = Relation(
			subj=concepts[subj_cui],
			rel_type=relation_types[rel_cui],
			obj=concepts[obj_cui]
		)
		relations.append(relation)
	return concepts, relation_types, relations


class UmlsDataset(Dataset):
	def __init__(self, relations):
		self.relations = relations

	def __len__(self):
		return len(self.relations)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		relations = self.relations[idx]

		return relations


class RelationCollator(object):
	def __init__(self, tokenizer, example_creator: RelationExampleCreator):
		self.tokenizer = tokenizer
		self.example_creator = example_creator

	def __call__(self, relations):
		# creates text examples
		examples = [self.example_creator.create(rel) for rel in relations]

		batch = self.tokenizer.batch_encode_plus(
			batch_text_or_text_pairs=examples,
			add_special_tokens=True,
			padding=True,
			return_tensors='pt',
			truncation=True,
			# TODO compute percentile
			max_length=64
		)

		return batch
