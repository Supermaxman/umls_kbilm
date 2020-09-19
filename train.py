
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, AdamW
from tqdm import tqdm
import os

from model_utils import KnowledgeBaseInfusedBert
from data_utils import RelationCollator, UmlsRelationDataset, load_umls, split_data, get_optimizer_params
from kb_utils import NameRelationExampleCreator


if __name__ == "__main__":
	seed = 0
	umls_directory = '/shared/hltdir1/disk1/home/max/data/ontologies/umls_2019/2019AA-full/2019AA/'
	data_folder = 'data'
	pre_model_name = 'monologg/biobert_v1.1_pubmed'
	save_directory = 'models/test'
	batch_size = 8
	weight_decay = 0.01
	learning_rate = 5e-5
	epochs = 100
	gamma = 6.0

	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)

	if torch.cuda.is_available():
		device = torch.device("cuda")
		print(f"Using GPU{torch.cuda.get_device_name(0)}")
	else:
		device = torch.device("cpu")
		print(f"Using CPU")

	print('Loading umls dataset')
	concepts, relation_types, relations = load_umls(umls_directory, data_folder)

	print('Loading model')
	tokenizer = BertTokenizer.from_pretrained(pre_model_name)
	config = BertConfig.from_pretrained(pre_model_name)
	config.gamma = gamma
	model = KnowledgeBaseInfusedBert.from_pretrained(pre_model_name, config=config)

	train_data, dev_data, test_data = split_data(relations)
	print(f'Train data size: {len(train_data)}')
	print(f'Dev data size: {len(dev_data)}')
	print(f'Test data size: {len(test_data)}')

	train_dataset = UmlsRelationDataset(train_data)
	dev_dataset = UmlsRelationDataset(dev_data)
	test_dataset = UmlsRelationDataset(test_data)

	example_creator = NameRelationExampleCreator()
	collator = RelationCollator(tokenizer, example_creator)

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
		num_workers=0,
		collate_fn=collator
	)
	test_dataloader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=0,
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
	if not os.path.exists(save_directory):
		os.mkdir(save_directory)

	for epoch in range(epochs):
		pbar = tqdm(train_dataloader)
		print(f"Initiating Epoch {epoch + 1}:")
		# Reset the total loss for each epoch.
		total_train_loss = 0
		train_loss_trajectory = list()

		# Reset timer for each epoch
		model.train()

		dev_log_frequency = 1
		n_steps = len(train_dataloader)
		dev_steps = int(n_steps / dev_log_frequency)
		for step, batch in enumerate(pbar):
			# Forward
			input_dict = {
				"input_ids": batch["input_ids"].to(device),
				"attention_mask": batch["attention_mask"].to(device),
				"pos_size": batch["pos_size"],
				"neg_size": batch["neg_size"],
				"total_size": batch["total_size"],
			}

			loss, pos_energy, neg_energy = model(**input_dict)
			# loss = loss / accumulation_steps
			# Accumulate loss
			total_train_loss += loss.item()

			# Backward: compute gradients
			loss.backward()

			avg_train_loss = total_train_loss / (step + 1)

			pbar.set_description(f"Epoch:{epoch + 1}|Batch:{step}/{len(train_dataloader)}|Avg. Loss:{avg_train_loss:.4f}|Loss:{loss.item():.4f}")

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
			if (step + 1) % dev_steps == 0:
				# Perform validation with the model and log the performance
				print("Running Validation...")
				# Put the model in evaluation mode--the dropout layers behave differently
				# during evaluation.
				model.eval()
				# TODO
				# dev_predicted_labels, dev_prediction_scores, dev_gold_labels = make_predictions_on_dataset(
				# 	dev_dataloader,
				# 	model,
				# 	device,
				# 	args.task + "_dev",
				# 	True
				# )
				# TP = 0
				# FP = 0
				# FN = 0
				#
				# print(f"Task:{micro_name:>15}\tN={TP + FN:.0f}\tF1={micro_f1:.4f}\tP={micro_p:.4f}\tR={micro_r:.4f}\tTP={TP:.0f}\tFP={FP:.0f}\tFN={FN:.0f}")

				# Put the model back in train setting
				model.train()

		# Calculate the average loss over all of the batches.
		avg_train_loss = total_train_loss / len(train_dataloader)

	model.save_pretrained(save_directory)
	tokenizer.save_pretrained(save_directory)
