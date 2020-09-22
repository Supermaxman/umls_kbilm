
from transformers import BertTokenizer
import os
import logging
import pytorch_lightning as pl

from model_utils import KnowledgeBaseInfusedBert
from data_utils import RelationCollator, UmlsRelationDataModule
from kb_utils import NameRelationExampleCreator


if __name__ == "__main__":
	# TODO parameterize below into config file for reproducibility
	seed = 0
	umls_directory = '/shared/hltdir1/disk1/home/max/data/ontologies/umls_2019/2019AA-full/2019AA/'
	data_folder = 'data'
	save_directory = 'models'
	model_name = 'umls-kbilm-v9'
	pre_model_name = 'monologg/biobert_v1.1_pubmed'
	# tpu_config = 'tpu_worker;0;10.225.43.138:8470'
	# os.environ['XRT_TPU_CONFIG'] = tpu_config
	weight_decay = 0.01
	learning_rate = 1e-5
	epochs = 100
	gamma = 24.0
	grad_norm_clip = 1.0
	max_seq_len = 64
	val_check_interval = 0.20
	is_distributed = True
	# batch_size = 64
	batch_size = 8
	negative_sample_size = 8
	accumulate_grad_batches = 1
	# accumulate_grad_batches = 4
	# amp_backend = 'native'
	amp_backend = 'native'
	precision = 32
	gpus = [4, 5, 6, 7]
	use_tpus = True
	tpu_cores = 8
	num_workers = 8 if use_tpus else 4

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
	dm = UmlsRelationDataModule(
		data_folder=data_folder,
		umls_directory=umls_directory,
		batch_size=batch_size,
		num_workers=num_workers,
		collator=collator
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
			progress_bar_refresh_rate=20,
			default_root_dir=save_directory,
			gradient_clip_val=grad_norm_clip,
			max_epochs=epochs,
			precision=precision,
			val_check_interval=val_check_interval,
			accumulate_grad_batches=accumulate_grad_batches
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
	trainer.fit(model, datamodule=dm)

	# TODO eval on test
	# logging.info('Evaluating...')
	# trainer.test(datamodule=dm)

