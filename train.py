
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, AdamW
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import logging
from transformers import BertModel
import pytorch_lightning as pl

from model_utils import KnowledgeBaseInfusedBert
from data_utils import RelationCollator, UmlsRelationDataset, load_umls, split_data, get_optimizer_params
from kb_utils import NameRelationExampleCreator


if __name__ == "__main__":
	seed = 0
	umls_directory = '/shared/hltdir1/disk1/home/max/data/ontologies/umls_2019/2019AA-full/2019AA/'
	data_folder = 'data'
	save_directory = 'models'
	log_directory = 'logs'
	model_name = 'umls-kbilm-v2'
	pre_model_name = 'monologg/biobert_v1.1_pubmed'
	batch_size = 8
	weight_decay = 0.01
	learning_rate = 1e-5
	epochs = 100
	gamma = 24.0
	grad_norm_clip = 1.0
	max_seq_len = 64
	dev_log_frequency = 10

	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	save_directory = os.path.join(save_directory, model_name)

	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	# Also add the stream handler so that it logs on STD out as well
	# Ref: https://stackoverflow.com/a/46098711/4535284
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)

	logfile = os.path.join(save_directory, "train_output.log")
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			logging.FileHandler(logfile, mode='w'),
			logging.StreamHandler()]
	)

	logging.info('Loading umls dataset')
	concepts, relation_types, relations = load_umls(umls_directory, data_folder)

	train_data, val_data, test_data = split_data(relations)
	logging.info(f'Train data size: {len(train_data)}')
	logging.info(f'Val data size: {len(val_data)}')
	logging.info(f'Test data size: {len(test_data)}')

	train_dataset = UmlsRelationDataset(train_data)
	val_dataset = UmlsRelationDataset(val_data)
	test_dataset = UmlsRelationDataset(test_data)

	example_creator = NameRelationExampleCreator()

	tokenizer = BertTokenizer.from_pretrained(pre_model_name)
	collator = RelationCollator(tokenizer, example_creator, max_seq_len)

	train_dataloader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=1,
		collate_fn=collator
	)
	val_dataloader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=1,
		collate_fn=collator
	)

	# test_dataloader = DataLoader(
	# 	test_dataset,
	# 	batch_size=batch_size,
	# 	shuffle=False,
	# 	num_workers=1,
	# 	collate_fn=collator
	# )

	logging.info('Loading model')
	bert = BertModel.from_pretrained(pre_model_name)
	model = KnowledgeBaseInfusedBert(bert, gamma, learning_rate, weight_decay)

	trainer = pl.Trainer(
		gpus=[4],
		default_root_dir=save_directory,
		gradient_clip_val=grad_norm_clip,
		max_epochs=epochs
	)
	trainer.fit(model, train_dataloader, val_dataloader)



