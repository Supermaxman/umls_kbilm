
from transformers import BertModel
from transformers import AdamW
from torch import nn
import torch
import pytorch_lightning as pl
import torch_xla.debug.metrics as met


class KnowledgeBaseInfusedBert(pl.LightningModule):
	def __init__(self, pre_model_name, gamma, learning_rate):
		super().__init__()
		self.bert = BertModel.from_pretrained(pre_model_name)
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.save_hyperparameters()

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
		# l1 norm
		energies = torch.abs(cls_embeddings).sum(dim=1)
		# [batch_size, sample_size]
		energies = energies.view(batch_size, sample_size)

		return energies

	def _energy_loss(self, energies):
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
		loss = batch_loss.mean()
		return pos_energies, neg_energies, neg_probs, loss

	def training_step(self, batch, batch_nb):
		energies = self(**batch)
		batch_size = energies.shape[0]
		pos_energies, neg_energies, neg_probs, loss = self._energy_loss(energies)

		neg_size = neg_energies.shape[1]
		pos_correct = (pos_energies.unsqueeze(1) < neg_energies).float()
		# # []
		exp_acc = (neg_probs * pos_correct).sum(dim=1).sum(dim=0) / batch_size
		uniform_acc = pos_correct.sum(dim=1).sum(dim=0) / neg_size
		# tensorboard_logs = {
		# 	'train_exp_acc': exp_acc,
		# 	'train_uniform_acc': uniform_acc
		# }
		# result = {'loss': loss, 'log': tensorboard_logs}

		# print(met.metrics_report())
		result = pl.TrainResult(loss)
		result.log('train_loss', loss)
		result.log('train_exp_acc', exp_acc)
		result.log('train_uniform_acc', uniform_acc)
		return result

	def validation_step(self, batch, batch_nb):
		energies = self(**batch)
		batch_size = energies.shape[0]
		pos_energies, neg_energies, neg_probs, loss = self._energy_loss(energies)

		neg_size = neg_energies.shape[1]
		pos_correct = (pos_energies.unsqueeze(1) < neg_energies).float()
		# []
		exp_acc = (neg_probs * pos_correct).sum(dim=1).sum(dim=0) / batch_size
		uniform_acc = pos_correct.sum(dim=1).sum(dim=0) / neg_size
		# first neg example replaces subj
		# pos_subj_uniform_correct = pos_correct[:, 0].sum(dim=0)
		# second neg example replaces obj
		# pos_obj_uniform_correct = pos_correct[:, 1].sum(dim=0)
		# pos_uniform_correct = pos_subj_uniform_correct + pos_obj_uniform_correct
		result = pl.EvalResult()
		result.log('val_loss', loss)
		result.log('val_exp_acc', exp_acc)
		result.log('val_uniform_acc', uniform_acc)
		# result = {'val_loss': loss}
		return result

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(
			self.parameters(),
			lr=self.learning_rate
		)
		return optimizer
