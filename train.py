
from transformers import BertTokenizer
import os
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from model_utils import KnowledgeBaseInfusedBert
from data_utils import RelationCollator, UmlsRelationDataModule, UmlsRelationDataset, load_umls, split_data
from kb_utils import NameRelationExampleCreator


if __name__ == "__main__":
	# TODO parameterize below into config file for reproducibility
	seed = 0
	umls_directory = '/shared/hltdir1/disk1/home/max/data/ontologies/umls_2019/2019AA-full/2019AA/'
	data_folder = 'data'
	save_directory = 'models'
	model_name = 'umls-kbilm-v13'
	pre_model_name = 'monologg/biobert_v1.1_pubmed'
	weight_decay = 0.01
	learning_rate = 1e-5
	epochs = 10
	gamma = 24.0
	grad_norm_clip = 1.0
	max_seq_len = 64
	val_check_interval = 0.20
	is_distributed = True
	# batch_size = 64
	batch_size = 2
	negative_sample_size = 8
	accumulate_grad_batches = 1
	# accumulate_grad_batches = 4
	# amp_backend = 'native'
	amp_backend = 'native'
	precision = 32
	gpus = [4, 5, 6, 7]
	use_tpus = True
	tpu_cores = 1
	num_workers = 1 if use_tpus else 4

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

	logging.info('Loading collator...')
	example_creator = NameRelationExampleCreator()
	tokenizer = BertTokenizer.from_pretrained(pre_model_name)
	# ensure negative_sample_size is correct based on batch_size
	collator = RelationCollator(
		tokenizer,
		example_creator,
		max_seq_len,
		negative_sample_size
	)

	logging.info('Loading dataset...')

	_, _, relations = load_umls(umls_directory, data_folder)
	train_data, val_data, _ = split_data(relations)
	train_dataset = UmlsRelationDataset(train_data)
	val_dataset = UmlsRelationDataset(val_data)

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

	# dm = UmlsRelationDataModule(
	# 	data_folder=data_folder,
	# 	umls_directory=umls_directory,
	# 	batch_size=batch_size,
	# 	num_workers=num_workers,
	# 	collator=collator
	# )

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
			# progress_bar_refresh_rate=1,
			default_root_dir=save_directory,
			# gradient_clip_val=grad_norm_clip,
			max_epochs=epochs,
			# precision=precision,
			# val_check_interval=val_check_interval,
			# num_sanity_val_steps=0,
			# accumulate_grad_batches=accumulate_grad_batches
		)
	else:
		trainer = pl.Trainer(
			gpus=gpus,
			default_root_dir=save_directory,
			gradient_clip_val=grad_norm_clip,
			max_epochs=epochs,
			precision=precision,
			distributed_backend='ddp' if is_distributed else 'dp',
			val_check_interval=val_check_interval,
			accumulate_grad_batches=accumulate_grad_batches,
			amp_backend=amp_backend
		)
	trainer.fit(model, train_dataloader, val_dataloader)

	# TODO eval on test
	# logging.info('Evaluating...')
	# trainer.test(datamodule=dm)

