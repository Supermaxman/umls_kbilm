
from transformers import BertTokenizer
import os
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from model_utils import KnowledgeBaseInfusedBert
from data_utils import RelationCollator, UmlsRelationDataset, load_umls, split_data
from kb_utils import NameRelationExampleCreator
from sample_utils import UniformNegativeSampler, BatchNegativeSampler


if __name__ == "__main__":
	# TODO parameterize below into config file for reproducibility
	seed = 0
	umls_directory = '/shared/hltdir1/disk1/home/max/data/ontologies/umls_2019/2019AA-full/2019AA/'
	data_folder = 'data'
	save_directory = 'models'
	model_name = 'umls-kbilm-v34'
	pre_model_name = 'monologg/biobert_v1.1_pubmed'
	learning_rate = 1e-5
	epochs = 10
	gamma = 12.0
	gradient_clip_val = 1.0
	weight_decay = 0.01
	max_seq_len = 64
	val_check_interval = 0.20
	is_distributed = True
	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
	# batch_size = 64
	batch_size = 8
	negative_sample_size = 16
	accumulate_grad_batches = 1
	# accumulate_grad_batches = 4
	precision = 32
	# gpus = [4, 5, 6, 7]
	gpus = [4]
	use_tpus = False
	tpu_cores = 8
	num_workers = 1

	pl.seed_everything(seed)

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

	logging.info('Loading dataset...')

	concepts, relation_types, relations = load_umls(umls_directory, data_folder)
	train_data, val_data, _ = split_data(relations)
	train_dataset = UmlsRelationDataset(train_data)
	val_dataset = UmlsRelationDataset(val_data)

	logging.info('Loading collator...')
	example_creator = NameRelationExampleCreator()
	# neg_sampler = UniformNegativeSampler(
	# 	list(concepts.values()),
	# 	negative_sample_size
	# )
	neg_sampler = BatchNegativeSampler(
		negative_sample_size
	)
	tokenizer = BertTokenizer.from_pretrained(pre_model_name)
	# ensure negative_sample_size is correct based on batch_size
	collator = RelationCollator(
		tokenizer,
		example_creator,
		neg_sampler,
		max_seq_len,
		force_max_seq_len=use_tpus
	)

	train_dataloader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=collator
	)
	val_dataloader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		collate_fn=collator
	)

	logging.info('Loading model...')
	model = KnowledgeBaseInfusedBert(
		pre_model_name=pre_model_name,
		gamma=gamma,
		learning_rate=learning_rate,
		weight_decay=weight_decay
	)

	logging.info('Training...')
	if use_tpus:
		trainer = pl.Trainer(
			tpu_cores=tpu_cores,
			default_root_dir=save_directory,
			max_epochs=epochs,
			precision=precision,
			val_check_interval=val_check_interval
		)
	else:
		if len(gpus) > 1:
			backend = 'ddp' if is_distributed else 'dp'
		else:
			backend = None
		trainer = pl.Trainer(
			gpus=gpus,
			default_root_dir=save_directory,
			max_epochs=epochs,
			precision=precision,
			val_check_interval=val_check_interval,
			distributed_backend=backend,
			gradient_clip_val=gradient_clip_val,
		)
	trainer.fit(model, train_dataloader, val_dataloader)

	# TODO eval on test
	# logging.info('Evaluating...')
	# trainer.test(datamodule=dm)

