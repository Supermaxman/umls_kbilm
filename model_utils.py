
from transformers import BertModel
from transformers import AdamW
from torch import nn
import torch
import pytorch_lightning as pl


class KnowledgeBaseInfusedBert(pl.LightningModule):
	def __init__(self, pre_model_name, gamma, adv_temp, learning_rate, weight_decay):
		super().__init__()
		self.bert = BertModel.from_pretrained(pre_model_name)
		self.gamma = gamma
		self.adv_temp = adv_temp
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
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
		energies = torch.norm(cls_embeddings, p=1, dim=1)
		# [batch_size, sample_size]
		energies = energies.view(batch_size, sample_size)

		return energies

	def _energy_loss(self, energies):
		# [batch_size]
		pos_energies = energies[:, 0]
		# [batch_size, neg_size]
		neg_energies = energies[:, 1:]
		with torch.no_grad():
			neg_probs = nn.Softmax(dim=1)(self.adv_temp * -neg_energies)
		pos_loss = -nn.LogSigmoid()(self.gamma - pos_energies)
		neg_loss = -neg_probs * nn.LogSigmoid()(neg_energies - self.gamma)
		neg_loss = neg_loss.sum(dim=1)
		loss = pos_loss + neg_loss
		return pos_energies, neg_energies, neg_probs, loss

	def training_step(self, batch, batch_nb):
		energies = self(**batch)
		batch_size = energies.shape[0]
		pos_energies, neg_energies, neg_probs, loss = self._energy_loss(energies)

		loss = loss.mean()

		neg_size = neg_energies.shape[1]
		pos_correct = (pos_energies.unsqueeze(1) < neg_energies).float()

		exp_acc = (neg_probs * pos_correct).sum(dim=1).sum(dim=0) / batch_size
		uniform_acc = pos_correct.sum(dim=1).sum(dim=0) / (batch_size * neg_size)

		result = {
			'loss': loss,
			'log': {
				'train_loss': loss,
				'train_exp_acc': exp_acc,
				'train_uniform_acc': uniform_acc
			}
		}
		return result

	def validation_step(self, batch, batch_nb):
		energies = self(**batch)
		pos_energies, neg_energies, neg_probs, loss = self._energy_loss(energies)

		neg_size = neg_energies.shape[1]
		pos_correct = (pos_energies.unsqueeze(1) < neg_energies).float()

		exp_acc = (neg_probs * pos_correct).sum(dim=1)
		uniform_acc = pos_correct.sum(dim=1) / neg_size

		result = {
			'val_loss': loss,
			'val_exp_acc': exp_acc,
			'val_uniform_acc': uniform_acc
		}
		print('--------validation_step---------')

		return result

	def validation_end(self, outputs):
		loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()
		exp_acc = torch.stack([x['val_exp_acc'] for x in outputs]).mean().item()
		uniform_acc = torch.stack([x['val_uniform_acc'] for x in outputs], dim=0).mean().item()
		print('--------validation_end---------')
		print(loss)
		print(type(loss))
		print('--------validation_end---------')

		result = {
			'val_loss': loss,
			'val_exp_acc': exp_acc,
			'val_uniform_acc': uniform_acc,
			'log': {
				'val_loss': loss,
				'val_exp_acc': exp_acc,
				'val_uniform_acc': uniform_acc
			}
		}
		return result

	def configure_optimizers(self):
		params = self._get_optimizer_params(self.weight_decay)
		optimizer = AdamW(
			params,
			lr=self.learning_rate,
			weight_decay=self.weight_decay,
			correct_bias=False
		)
		return optimizer

	def _get_optimizer_params(self, weight_decay):
		param_optimizer = list(self.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_params = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
			 'weight_decay': weight_decay},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		return optimizer_params
