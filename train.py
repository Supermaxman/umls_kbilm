
from transformers import BertTokenizer
import os
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.callbacks import ModelCheckpoint

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
	model_name = 'umls-kbilm-t1'
	pre_model_name = 'monologg/biobert_v1.1_pubmed'
	learning_rate = 1e-5
	epochs = 2
	#  {3, 6, 9, 12, 18, 24, 30}
	gamma = 24.0
	#  {0.5, 1.0}
	adv_temp = 1.0
	gradient_clip_val = 1.0
	weight_decay = 0.01
	max_seq_len = 64
	val_check_interval = 0.50
	is_distributed = True
	# export TPU_IP_ADDRESS=10.155.6.34
	# export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
	# batch_size = 64
	batch_size = 16
	negative_sample_size = 16
	accumulate_grad_batches = 1
	# accumulate_grad_batches = 4
	gpus = [3, 4, 6, 7]
	# gpus = [4]
	use_tpus = True
	precision = 16 if use_tpus else 32
	tpu_cores = 8
	num_workers = 1
	deterministic = False

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

	logging.warning('Testing system with only 1000 examples!!!')
	relations = relations[:1000]
	concept_list = list(concepts.values())
	train_data, val_data, _ = split_data(relations)
	train_relations_set = set(train_data)
	train_dataset = UmlsRelationDataset(train_data)
	val_dataset = UmlsRelationDataset(val_data)

	callbacks = []
	logging.info('Loading collator...')
	example_creator = NameRelationExampleCreator()
	# train_neg_sampler = BatchNegativeSampler(
	# 	negative_sample_size
	# )
	# val_neg_sampler = train_neg_sampler
	train_neg_sampler = UniformNegativeSampler(
		concept_list,
		train_relations_set,
		negative_sample_size,
		seed=seed,
		train_callback=True
	)
	callbacks.append(train_neg_sampler)
	val_neg_sampler = UniformNegativeSampler(
		concept_list,
		train_relations_set,
		negative_sample_size,
		seed=seed,
		val_callback=True
	)
	callbacks.append(val_neg_sampler)
	tokenizer = BertTokenizer.from_pretrained(pre_model_name)
	# ensure negative_sample_size is correct based on batch_size
	train_collator = RelationCollator(
		tokenizer,
		example_creator,
		train_neg_sampler,
		max_seq_len,
		force_max_seq_len=use_tpus
	)
	train_dataloader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=train_collator
	)

	val_collator = RelationCollator(
		tokenizer,
		example_creator,
		val_neg_sampler,
		max_seq_len,
		force_max_seq_len=use_tpus
	)
	val_dataloader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		collate_fn=val_collator
	)

	logging.info('Loading model...')
	model = KnowledgeBaseInfusedBert(
		pre_model_name=pre_model_name,
		gamma=gamma,
		adv_temp=adv_temp,
		learning_rate=learning_rate,
		weight_decay=weight_decay
	)

	checkpoint_callback = ModelCheckpoint(
		verbose=True,
		monitor='val_loss',
		save_last=True,
		save_top_k=2,
		mode='min'
	)

	logging.info('Training...')
	if use_tpus:
		logging.warning('Gradient clipping slows down TPU training drastically, disabled for now.')
		trainer = pl.Trainer(
			tpu_cores=tpu_cores,
			default_root_dir=save_directory,
			max_epochs=epochs,
			precision=precision,
			val_check_interval=val_check_interval,
			deterministic=deterministic,
			callbacks=callbacks,
			checkpoint_callback=checkpoint_callback
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
			deterministic=deterministic,
			callbacks=callbacks,
			checkpoint_callback=checkpoint_callback
		)
	trainer.fit(model, train_dataloader, val_dataloader)

	# TODO eval on test
	# logging.info('Evaluating...')
	# trainer.test(datamodule=dm)

