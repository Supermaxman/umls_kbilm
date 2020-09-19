
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, AdamW
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import logging

from model_utils import KnowledgeBaseInfusedBert
from data_utils import RelationCollator, UmlsRelationDataset, load_umls, split_data, get_optimizer_params
from kb_utils import NameRelationExampleCreator


if __name__ == "__main__":
	seed = 0
	umls_directory = '/shared/hltdir1/disk1/home/max/data/ontologies/umls_2019/2019AA-full/2019AA/'
	data_folder = 'data'
	save_directory = 'models'
	log_directory = 'logs'
	model_name = 'v1'
	pre_model_name = 'monologg/biobert_v1.1_pubmed'
	batch_size = 8
	weight_decay = 0.01
	learning_rate = 5e-5
	epochs = 100
	gamma = 6.0
	max_seq_len = 64
	dev_log_frequency = 100

	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	save_directory = os.path.join(save_directory, model_name)
	log_directory = os.path.join(log_directory, model_name)

	if not os.path.exists(save_directory):
		os.mkdir(save_directory)
	if not os.path.exists(log_directory):
		os.mkdir(log_directory)

	# Also add the stream handler so that it logs on STD out as well
	# Ref: https://stackoverflow.com/a/46098711/4535284
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)

	logfile = os.path.join(log_directory, "train_output.log")
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			logging.FileHandler(logfile, mode='w'),
			logging.StreamHandler()]
	)

	if torch.cuda.is_available():
		device = torch.device("cuda")
		logging.info(f"Using GPU{torch.cuda.get_device_name(0)}")
	else:
		device = torch.device("cpu")
		logging.info(f"Using CPU")

	logging.info('Loading umls dataset')
	concepts, relation_types, relations = load_umls(umls_directory, data_folder)

	logging.info('Loading model')
	tokenizer = BertTokenizer.from_pretrained(pre_model_name)
	config = BertConfig.from_pretrained(pre_model_name)
	config.batch_size = batch_size
	config.weight_decay = weight_decay
	config.learning_rate = learning_rate
	config.epochs = epochs
	config.gamma = gamma
	config.max_seq_len = max_seq_len
	config.model_name = model_name
	config.pre_model_name = pre_model_name
	config.seed = seed

	model = KnowledgeBaseInfusedBert.from_pretrained(pre_model_name, config=config)
	model.to(device)

	train_data, dev_data, test_data = split_data(relations)
	logging.info(f'Train data size: {len(train_data)}')
	logging.info(f'Dev data size: {len(dev_data)}')
	logging.info(f'Test data size: {len(test_data)}')

	train_dataset = UmlsRelationDataset(train_data)
	dev_dataset = UmlsRelationDataset(dev_data)
	test_dataset = UmlsRelationDataset(test_data)

	example_creator = NameRelationExampleCreator()
	collator = RelationCollator(tokenizer, example_creator, max_seq_len)

	train_dataloader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=1,
		collate_fn=collator
	)
	dev_dataloader = DataLoader(
		dev_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=1,
		collate_fn=collator
	)
	test_dataloader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=1,
		collate_fn=collator
	)
	params = get_optimizer_params(model, weight_decay)
	# TODO get warm up / decay schedule code
	optimizer = AdamW(
		params,
		lr=learning_rate,
		weight_decay=weight_decay,
		correct_bias=False
	)

	writer = SummaryWriter(log_directory)
	for epoch in range(epochs):
		pbar = tqdm(train_dataloader)
		logging.info(f"Initiating Epoch {epoch + 1}:")
		# Reset the total loss for each epoch.
		total_train_loss = 0.0

		# Reset timer for each epoch
		model.train()

		n_steps = len(train_dataloader)
		dev_steps = int(n_steps / dev_log_frequency)
		for step, batch in enumerate(pbar):
			if step == 0:
				logging.info(f'pos_size={batch["pos_size"]}')
				logging.info(f'neg_size={batch["neg_size"]}')
				logging.info(f'total_size={batch["total_size"]}')
				logging.info(f'input_ids={batch["input_ids"].shape}')
			# Forward
			input_dict = {
				"input_ids": batch["input_ids"].to(device),
				"attention_mask": batch["attention_mask"].to(device),
				"pos_size": batch["pos_size"],
				"neg_size": batch["neg_size"],
				"total_size": batch["total_size"],
			}

			results = model(**input_dict)
			loss = results['loss']
			# loss = loss / accumulation_steps
			# Accumulate loss
			loss_value = loss.item()
			total_train_loss += loss_value

			# Backward: compute gradients
			loss.backward()

			avg_train_loss = total_train_loss / (step + 1)

			pbar.set_description(f"Epoch:{epoch + 1}|Batch:{step}/{len(train_dataloader)}|Avg. Loss:{avg_train_loss:.4f}|Loss:{loss_value:.4f}")
			writer.add_scalar('train/train_loss', loss_value, step)
			# Clip the norm of the gradients to 1.0.
			# This is to help prevent the "exploding gradients" problem.
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			# Update parameters
			optimizer.step()

			# Clean the model's previous gradients
			model.zero_grad()  # Reset gradients tensors

			# Update the learning rate.
			# scheduler.step()
			pbar.update()
			if (step + 1) % dev_steps == 0 or step == 0:
				# Perform validation with the model and log the performance
				logging.info("Running Validation...")
				# Put the model in evaluation mode--the dropout layers behave differently
				# during evaluation.
				model.eval()
				dev_pbar = tqdm(dev_dataloader)
				total_count = 0
				total_exp_correct = 0.0
				total_subj_uni_correct = 0.0
				total_obj_uni_correct = 0.0
				total_dev_loss = 0.0
				with torch.no_grad():
					for _, dev_batch in enumerate(dev_pbar):
						# Create testing instance for model
						input_dict = {
							"input_ids": dev_batch["input_ids"].to(device),
							"attention_mask": dev_batch["attention_mask"].to(device),
							"pos_size": dev_batch["pos_size"],
							"neg_size": dev_batch["neg_size"],
							"total_size": dev_batch["total_size"],
						}
						# dev_loss [pos_size]
						# dev_pos_energy [pos_size]
						# dev_neg_energy [pos_size, neg_size]
						# dev_neg_probs [pos_size, neg_size]
						dev_results = model(**input_dict)
						total_count += dev_batch["pos_size"]
						total_dev_loss += dev_results['loss'].sum().item()
						# [pos_size, neg_size]
						dev_correct = (dev_results['dev_pos_energy'].unsqueeze(1) < dev_results['dev_neg_energy']).float()
						# []
						dev_exp_correct = (dev_results['neg_probs'] * dev_correct).sum(dim=1).sum(dim=0).item()
						total_exp_correct += dev_exp_correct
						# first neg example replaces subj
						dev_subj_uniform_correct = dev_correct[:, 0].sum(dim=0).item()
						total_subj_uni_correct += dev_subj_uniform_correct
						# second neg example replaces obj
						dev_obj_uniform_correct = dev_correct[:, 1].sum(dim=0).item()
						total_obj_uni_correct += dev_obj_uniform_correct

				dev_subj_uni_acc = total_subj_uni_correct / total_count
				dev_obj_uni_acc = total_obj_uni_correct / total_count
				dev_uni_acc = (total_subj_uni_correct + total_obj_uni_correct) / (2 * total_count)
				dev_exp_acc = total_exp_correct / total_count
				dev_loss = total_dev_loss / total_count
				writer.add_scalar('dev/dev_loss', dev_loss, step)
				writer.add_scalar('dev/dev_exp_acc', dev_exp_acc, step)
				writer.add_scalar('dev/dev_uni_acc', dev_uni_acc, step)
				writer.add_scalar('dev/dev_subj_uni_acc', dev_subj_uni_acc, step)
				writer.add_scalar('dev/dev_obj_uni_acc', dev_obj_uni_acc, step)

				logging.info(f"DEV:\tLoss={dev_loss:.4f}\tExpected Accuracy={dev_exp_acc:.4f}\tUniform Accuracy={dev_uni_acc:.4f}")
				logging.info('Saving model...')
				config.save_pretrained(save_directory)
				model.save_pretrained(save_directory)
				tokenizer.save_pretrained(save_directory)
				# Put the model back in train setting
				model.train()

		# Calculate the average loss over all of the batches.
		avg_train_loss = total_train_loss / len(train_dataloader)
	config.save_pretrained(save_directory)
	model.save_pretrained(save_directory)
	tokenizer.save_pretrained(save_directory)
	writer.close()

