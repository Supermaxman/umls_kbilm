
from transformers import BertModel
from transformers import AdamW
from torch import nn
import torch
import pytorch_lightning as pl


class KnowledgeBaseInfusedBert(pl.LightningModule):
	def __init__(self, pre_model_name, gamma, learning_rate, weight_decay):
		super().__init__()

		self.bert = BertModel.from_pretrained(pre_model_name)
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay

	def forward(self, input_ids, attention_mask):
		batch_size, sample_size, max_seq_len = input_ids.shape
		# [batch_size * sample_size, max_seq_len]
		input_ids = input_ids.view(batch_size * sample_size, max_seq_len)
		# [batch_size * sample_size, max_seq_len]
		attention_mask = attention_mask.view(batch_size * sample_size, max_seq_len)
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask
		)
		contextualized_embeddings = outputs[0]
		# [total_size, hidden_size]
		cls_embeddings = contextualized_embeddings[:, 0]
		# energies = self.energy_linear(cls_embeddings)
		# [total_size]
		energies = torch.norm(cls_embeddings, p=1, dim=1)
		# [batch_size, sample_size]
		energies = energies.view(batch_size, sample_size)

		return energies

	def training_step(self, batch, batch_nb):
		energies = self(**batch)
		batch_size = energies.shape[0]
		# [batch_size]
		pos_energies = energies[:, 0]
		# [batch_size, neg_size]
		neg_energies = energies[:, 1:]
		with torch.no_grad():
			neg_probs = nn.Softmax(dim=1)(-neg_energies)
		pos_loss = -nn.LogSigmoid()(self.gamma - pos_energies)
		neg_loss = -neg_probs * nn.LogSigmoid()(neg_energies - self.gamma)
		neg_loss = neg_loss.sum(dim=1)
		batch_loss = pos_loss + neg_loss
		# TODO determine if i should mean here or not
		loss = batch_loss.mean()

		pos_correct = (pos_energies.unsqueeze(1) < neg_energies).float()
		# []
		pos_exp_correct = (neg_probs * pos_correct).sum(dim=1).sum(dim=0)
		# first neg example replaces subj
		pos_subj_uniform_correct = pos_correct[:, 0].sum(dim=0)
		# second neg example replaces obj
		pos_obj_uniform_correct = pos_correct[:, 1].sum(dim=0)
		pos_uniform_correct = pos_subj_uniform_correct + pos_obj_uniform_correct

		tensorboard_logs = {
			'train_loss': loss,
			'train_exp_acc': pos_exp_correct / batch_size,
			'train_uniform_acc': pos_uniform_correct / (2 * batch_size),
		}
		return {'loss': loss, 'log': tensorboard_logs}

	def validation_step(self, batch, batch_nb):
		energies = self(**batch)
		batch_size = energies.shape[0]
		# [batch_size]
		pos_energies = energies[:, 0]
		# [batch_size, neg_size]
		neg_energies = energies[:, 1:]
		with torch.no_grad():
			neg_probs = nn.Softmax(dim=1)(-neg_energies)
		pos_loss = -nn.LogSigmoid()(self.gamma - pos_energies)
		neg_loss = -neg_probs * nn.LogSigmoid()(neg_energies - self.gamma)
		neg_loss = neg_loss.sum(dim=1)
		batch_loss = pos_loss + neg_loss
		# TODO determine if i should mean here or not
		loss = batch_loss.mean()

		pos_correct = (pos_energies.unsqueeze(1) < neg_energies).float()
		# []
		pos_exp_correct = (neg_probs * pos_correct).sum(dim=1).sum(dim=0)
		# first neg example replaces subj
		pos_subj_uniform_correct = pos_correct[:, 0].sum(dim=0)
		# second neg example replaces obj
		pos_obj_uniform_correct = pos_correct[:, 1].sum(dim=0)
		pos_uniform_correct = pos_subj_uniform_correct + pos_obj_uniform_correct

		tensorboard_logs = {
			'val_loss': loss,
			'val_exp_acc': pos_exp_correct / batch_size,
			'val_subj_uniform_acc': pos_subj_uniform_correct / batch_size,
			'val_obj_uniform_acc': pos_obj_uniform_correct / batch_size,
			'val_uniform_acc': pos_uniform_correct / (2 * batch_size),
		}
		return {'loss': loss, 'log': tensorboard_logs}

	def configure_optimizers(self):
		params = self.get_optimizer_params(self.weight_decay)
		optimizer = AdamW(
			params,
			lr=self.learning_rate,
			weight_decay=self.weight_decay,
			correct_bias=False
		)
		return optimizer

	def get_optimizer_params(self, weight_decay):
		param_optimizer = list(self.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_params = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
			 'weight_decay': weight_decay},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		return optimizer_params
