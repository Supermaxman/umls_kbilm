
from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch


class KnowledgeBaseInfusedBert(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.bert = BertModel(config)
		self.gamma = config.gamma

		# self.energy_linear = nn.Linear(config.hidden_size, 1)

		self.init_weights()

	def forward(
			self,
			input_ids,
			attention_mask=None,
			token_type_ids=None,
			position_ids=None,
			head_mask=None,
			inputs_embeds=None,
			labels=None,
			pos_size=None,
			neg_size=None,
			total_size=None
	):

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		contextualized_embeddings = outputs[0]
		# [total_size, hidden_size]
		cls_embeddings = contextualized_embeddings[:, 0]
		# energies = self.energy_linear(cls_embeddings)
		# [total_size]
		energies = torch.norm(cls_embeddings, p=1, dim=1)
		# [pos_size]
		pos_energies = energies[:pos_size].view(pos_size)
		# [pos_size, neg_size]
		neg_energies = energies[pos_size:total_size].view(pos_size, neg_size)
		with torch.no_grad():
			neg_probs = nn.Softmax(dim=1)(-neg_energies)
		pos_loss = -nn.LogSigmoid()(self.gamma - pos_energies)
		neg_loss = -neg_probs * nn.LogSigmoid()(neg_energies - self.gamma)
		neg_loss = neg_loss.sum(dim=1)
		batch_loss = pos_loss + neg_loss
		loss = batch_loss.mean()

		pos_correct = (pos_energies.unsqueeze(1) < neg_energies).float()
		# []
		pos_exp_correct = (neg_probs * pos_correct).sum(dim=1).sum(dim=0)
		# first neg example replaces subj
		pos_subj_uniform_correct = pos_correct[:, 0].sum(dim=0)
		# second neg example replaces obj
		pos_obj_uniform_correct = pos_correct[:, 1].sum(dim=0)
		pos_uniform_correct = pos_subj_uniform_correct + pos_obj_uniform_correct

		results = {
			'loss': loss,
			'batch_loss': batch_loss,
			'pos_energies': pos_energies,
			'neg_energies': neg_energies,
			'neg_probs': neg_probs,
			'pos_correct': pos_correct,
			'pos_exp_correct': pos_exp_correct,
			'pos_subj_uniform_correct': pos_subj_uniform_correct,
			'pos_obj_uniform_correct': pos_obj_uniform_correct,
			'pos_uniform_correct': pos_uniform_correct,
		}

		return results

