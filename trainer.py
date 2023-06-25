import torch
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import random
import time
import math
import os
import pickle
import numpy as np
import wandb

save_folder = './params/'

max_length = 512
sample_limit = 10

margin = 9

class Trainer:
	def __init__(self, data_loader, model, tokenizer, optimizer, scheduler, device, hyperparams):

		self.data_loader = data_loader
		self.model = model

		self.tokenizer = tokenizer
		self.optimizer = optimizer
		self.device = device
		self.identifier = hyperparams['identifier']
		self.hyperparams = hyperparams
		self.save_folder = save_folder
		self.load_epoch = hyperparams['load_epoch']

		self.scheduler = scheduler

		model.to(device)

		self.result_log = self.save_folder + self.identifier + '.txt'
		self.param_path_template = self.save_folder + self.identifier + '-epc_{0}_metric_{1}'  + '.pt'
		self.history_path = self.save_folder + self.identifier + '-history_{0}'  + '.pkl'


		self.best_metric = {'acc': 0, 'f1': 0, 
			'raw_mrr': 0, 'raw_hits1': 0, 'raw_hits3': 0, 'raw_hits10': 0,
			'fil_mr': 100000000000, 'fil_mrr': 0, 'fil_hits1': 0, 'fil_hits3': 0, 'fil_hits10': 0,
		}

		self.best_epoch = {'acc': -1, 'f1': -1, 
			'raw_mrr': -1, 'raw_hits1': -1, 'raw_hits3': -1, 'raw_hits10': -1,
			'fil_mr': -1, 'fil_mrr': -1, 'fil_hits1': -1, 'fil_hits3': -1, 'fil_hits10': -1,
		}

		self.history_value = {'acc': [], 'f1': [], 
			'raw_mrr': [], 'raw_hits1': [], 'raw_hits3': [], 'raw_hits10': [],
			'fil_mr': [], 'fil_mrr': [], 'fil_hits1': [], 'fil_hits3': [], 'fil_hits10': [],
		}


		if not os.path.exists(save_folder):
			os.makedirs(save_folder)


		load_path = hyperparams['load_path']
		if load_path == None and self.load_epoch >= 0:
			load_path = self.param_path_template.format(self.load_epoch, hyperparams['load_metric'])
			history_path = self.history_path.format(self.load_epoch)
			if os.path.exists(history_path):
				with open(history_path, 'rb') as fil:
					self.history_value = pickle.load(fil)

		if load_path != None:
			if not (load_path.startswith(save_folder) or load_path.startswith('./saveparams/')):
				load_path = save_folder + load_path

			if os.path.exists(load_path):
				
				try:
					checkpoint = torch.load(load_path)
					model.load_state_dict(checkpoint['model_state_dict'], strict=False)
				
					optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
					scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
					
					print('Model & Optimizer  Parameters loaded from {0}.'.format(load_path))
				except:
					model.load_state_dict(torch.load(load_path), strict=False)
					print('Parameters loaded from {0}.'.format(load_path))
			else:
				print('Parameters {0} Not Found'.format(load_path))

		self.load_path = load_path
		import signal
		signal.signal(signal.SIGINT, self.debug_signal_handler)

	def run(self):
		self.train()

	def train(self):
		model = self.model
		tokenizer = self.tokenizer
		optimizer = self.optimizer
		scheduler = self.scheduler

		device = self.device
		hyperparams = self.hyperparams

		batch_size = hyperparams['batch_size'] 
		epoch = hyperparams['epoch'] 
		neg_rate = hyperparams['neg_rate']

		data_loader = self.data_loader
		ent2id = data_loader.ent2id
		rel2id = data_loader.rel2id
		entity_list = data_loader.entity_list
		relation_list = data_loader.relation_list
		groundtruth = data_loader.get_groundtruth()

		# criterion 
		criterion = torch.nn.CrossEntropyLoss()
		bce_criterion = torch.nn.BCELoss(reduction='none')
		sigmoid = torch.nn.Sigmoid()

		model.train()

		if hyperparams['wandb']:
			wandb.init(
				project="lmke",
				name=self.identifier,
				config=hyperparams
			)

		degrees = data_loader.statistics['degrees']

		if hyperparams['task'] == 'LP':
			modes = ["link_prediction_h", "link_prediction_t"]
		else:
			modes = ['triple_classification']

		for epc in range(self.load_epoch+1, epoch):
			total_loss_triple_classification = 0
			total_loss_link_prediction = 0
			total_klloss = 0

			total_accuracy = 0 # accuracy of triple classification
			total_accuracy_reverse = 0

			total_pos_acc = 0
			total_neg_acc = 0
			total_hits1 = { i: 0 for i in ['h', 'r', 't']}

			total_bce_acc = { i: 0 for i in ['h', 'r', 't']}
			total_bce_acc_pos = { i: 0 for i in ['h', 'r', 't']}
			total_bce_acc_neg = { i: 0 for i in ['h', 'r', 't']}

			avg_bce_acc = { i: 0 for i in ['h', 'r', 't']}
			avg_bce_acc_pos = { i: 0 for i in ['h', 'r', 't']}
			avg_bce_acc_neg = { i: 0 for i in ['h', 'r', 't']}

			avg_hits1 = { i: 0 for i in ['h', 'r', 't']}

			time_begin = time.time()
			
			data_sampler = data_loader.train_data_sampler()
			n_batch = len(data_sampler)
			dataset_size = data_sampler.get_dataset_size()

			real_dataset_size = dataset_size / (1+neg_rate)

			for i_b, batch in tqdm(enumerate(data_sampler), total=n_batch):
				triples = [i[0] for i in batch]
				triple_degrees = [ [degrees[e] for e in triple] for triple in triples]
				batch_size_ = len(batch)

				real_idxs = [ _ for _, i in enumerate(batch) if i[1] == 1]
				real_triples = [ i[0] for _, i in enumerate(batch) if i[1] == 1]
				real_triple_degrees = [ [degrees.get(e, 0) for e in triple] for triple in real_triples]

				real_batch_size = len(real_triples)


				for mode in  modes:
					if mode == 'triple_classification':
						inputs, positions = data_loader.batch_tokenize(triples, mode)
						inputs.to(device)	

						labels = [i[1] for i in batch]
						labels = torch.tensor(labels).to(device)

						preds = model(inputs, positions, mode)
						loss = criterion(preds, labels) 

						pred_labels = preds.argmax(dim=1)
						total_accuracy += (pred_labels == labels).int().sum().item()

						total_loss_triple_classification += loss.item() * batch_size_

					elif mode in ["link_prediction_h", "link_prediction_r", "link_prediction_t"]:
						real_inputs, real_positions = data_loader.batch_tokenize(real_triples, mode)
						real_inputs.to(device)

						if hyperparams['contrastive']:
							target_inputs, target_positions = data_loader.batch_tokenize_target(real_triples, mode)
							target_inputs.to(device)

						label_idx_list = []

						if hyperparams['contrastive']:
							labels = torch.zeros((len(real_triples), len(real_triples))).to(device)
							if mode == "link_prediction_h":
								targets = [ triple[0] for triple in real_triples]
								target_idxs = [ ent2id[tar] for tar in targets]
								for i, triple in enumerate(real_triples):
									h, r, t = triple
									expects = set(groundtruth['train']['head'][(r, t)])
									label_idx = [ i_t for i_t, target in enumerate(targets) if target in expects] 
									label_idx_list.append(label_idx)
									labels[i, label_idx] = 1 
							elif mode == "link_prediction_r":
								targets = [ triple[1] for triple in real_triples]
								target_idxs = [ rel2id[tar] for tar in targets]
								for i, triple in enumerate(real_triples):
									h, r, t = triple
									expects = set(groundtruth['train']['rel'][(h, t)])
									label_idx = [ i_t for i_t, target in enumerate(targets) if target in expects] 
									label_idx_list.append(label_idx)
									labels[i, label_idx] = 1 
							elif mode == "link_prediction_t":
								targets = [ triple[2] for triple in real_triples]
								target_idxs = [ ent2id[tar] for tar in targets]
								for i, triple in enumerate(real_triples):
									h, r, t = triple
									expects = set(groundtruth['train']['tail'][(r, h)])
									label_idx = [ i_t for i_t, target in enumerate(targets) if target in expects] 
									label_idx_list.append(label_idx)
									labels[i, label_idx] = 1
							candidate_degrees = [ degrees.get(tar, 0) for tar in targets] 

						else:
							if mode == "link_prediction_h":
								labels = torch.zeros((len(real_triples), len(ent2id))).to(device)
								candidate_degrees = [ degrees.get(tar, 0) for tar in entity_list]
								for i, triple in enumerate(real_triples):
									h, r, t = triple
									label = groundtruth['train']['head'][(r, t)]
									label_idx = [ ent2id[l] for l in label]
									label_idx_list.append(label_idx)
									labels[i, label_idx] = 1 
							elif mode == "link_prediction_r":
								labels = torch.zeros((len(real_triples), len(rel2id))).to(device)
								candidate_degrees = [ degrees.get(tar, 0) for tar in relation_list]
								for i, triple in enumerate(real_triples):
									h, r, t = triple
									label = groundtruth['train']['rel'][(h, t)]
									label_idx = [ rel2id[l] for l in label]
									label_idx_list.append(label_idx)
									labels[i, label_idx] = 1 
							elif mode == "link_prediction_t":
								labels = torch.zeros((len(real_triples), len(ent2id))).to(device)
								candidate_degrees = [ degrees.get(tar, 0) for tar in entity_list]
								for i, triple in enumerate(real_triples):
									h, r, t = triple
									label = groundtruth['train']['tail'][(r, h)]
									label_idx = [ ent2id[l] for l in label]
									label_idx_list.append(label_idx)
									labels[i, label_idx] = 1 
							
						loss = 0

						bce_loss = []
						preds_list = []

						if not hyperparams['no_use_lm']:
							if hyperparams['contrastive']:
								target_preds = model(real_inputs, real_positions, mode)
								target_encodes = model.encode_target(target_inputs, target_positions, mode)

								preds = model.match(target_preds, target_encodes, real_triple_degrees, mode)
								
							else:
								preds, confidence = model(real_inputs, real_positions, mode)
								if hyperparams['rdrop']:
									preds1 = preds[:real_batch_size // 2]
									preds1 = preds1.resize(preds1.shape[0] * preds1.shape[1], 1)
									preds1 = torch.cat([preds1, torch.zeros(preds1.shape).to(device)], dim=-1)

									preds2 = preds[real_batch_size // 2: ]
									preds2 = preds2.resize(preds2.shape[0] * preds2.shape[1], 1)
									preds2 = torch.cat([preds2, torch.zeros(preds2.shape).to(device)], dim=-1)

									KL = torch.nn.KLDivLoss()
									#kl_loss = ((preds1 * preds1.log() - preds1 * preds2.log())).mean() + ((preds2 * preds2.log() - preds2 * preds1.log())).mean()
									kl_loss = F.kl_div(preds1.log_softmax(dim=-1), preds2.softmax(dim=-1), reduction='mean') + F.kl_div(preds2.log_softmax(dim=-1), preds1.softmax(dim=-1), reduction='mean')
									loss += kl_loss * 10
									total_klloss += kl_loss.item()

							preds = sigmoid(preds)
							bce_loss.append(bce_criterion(preds, labels))
							preds_list.append(preds)

							
						if hyperparams['use_structure']:
							triple_score = model.forward_transe(real_positions, mode)	
							preds_transe = (margin - triple_score).sigmoid()

							if hyperparams['contrastive']:
								preds_transe = preds_transe[:, target_idxs]
							
							bce_loss.append(bce_criterion(preds_transe, labels))
							preds_list.append(preds_transe)

							preds = preds_transe

												
						pred_labels = (preds > 0.5).int() # 二分类

						for i in range(real_batch_size):
							
							pos_idx = sorted(label_idx_list[i])
							pos_set = set(pos_idx)
							neg_idx = [ _ for _ in range(labels.shape[1]) if not _ in pos_set]
							
							# Code for Exp of 'Importance of Negative Sampling Size'
							'''
							neg_sampling_size = 128
							other_idx = [ _ for _ in range(labels.shape[1]) if _ != i]
							other_idx = random.sample(other_idx, neg_sampling_size)

							neg_idx = [ _ for _ in other_idx if not _ in pos_set]
							other_pos_idx = [ _ for _ in other_idx if _ in pos_set]
							pos_idx = [i] + other_pos_idx

							assert ( len(pos_idx+neg_idx) == 1 + neg_sampling_size)
							assert ( i in pos_idx)
							'''
							
							for j, bl in enumerate(bce_loss):
								# separately add lm_loss, transe_loss, and ensembled_loss
								l = bl[i]
								pos_loss = l[pos_idx].mean()

								if hyperparams['self_adversarial']:
									# self-adversarial sampling
									neg_selfadv_weight = preds_list[j][i][neg_idx] #selfadv_weight[i][neg_idx]
									neg_weights = neg_selfadv_weight.softmax(dim=-1)
									neg_loss = (l[neg_idx]*neg_weights).sum()
								else:
									neg_loss = l[neg_idx].mean()

								loss += pos_loss + neg_loss


							total_bce_acc_pos[mode[-1]] += (pred_labels[i] == labels[i])[pos_idx].int().sum().item() / max(len(pos_idx), 1)
							total_bce_acc_neg[mode[-1]] += (pred_labels[i] == labels[i])[neg_idx].int().sum().item() / max(len(neg_idx), 1)
						
								
						total_bce_acc[mode[-1]] += (pred_labels == labels).int().sum().item() / labels.shape[1]
						
						total_loss_link_prediction += loss.item() 

					
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					scheduler.step()



			avg_loss_triple_classification = total_loss_triple_classification / dataset_size
			avg_klloss = total_klloss / dataset_size
			avg_accuracy = total_accuracy / dataset_size
			avg_accuracy_reverse = total_accuracy_reverse / dataset_size

			avg_hits1['h'] = total_hits1['h'] / real_dataset_size 
			avg_hits1['r'] = total_hits1['r'] / real_dataset_size 
			avg_hits1['t'] = total_hits1['t'] / real_dataset_size 
			avg_bce_acc['h'] = total_bce_acc['h'] / real_dataset_size 
			avg_bce_acc['r'] = total_bce_acc['r'] / real_dataset_size 
			avg_bce_acc['t'] = total_bce_acc['t'] / real_dataset_size 
			avg_bce_acc_pos['h'] = total_bce_acc_pos['h'] / real_dataset_size 
			avg_bce_acc_pos['r'] = total_bce_acc_pos['r'] / real_dataset_size 
			avg_bce_acc_pos['t'] = total_bce_acc_pos['t'] / real_dataset_size 
			avg_bce_acc_neg['h'] = total_bce_acc_neg['h'] / real_dataset_size 
			avg_bce_acc_neg['r'] = total_bce_acc_neg['r'] / real_dataset_size 
			avg_bce_acc_neg['t'] = total_bce_acc_neg['t'] / real_dataset_size 

			avg_loss_link_prediction = total_loss_link_prediction / real_dataset_size

			time_end = time.time()
			time_epoch = time_end - time_begin
			print('Train: Epoch: {} , Avg_Triple_Classification_Loss: {}, avg_klloss: {},  Avg_Accuracy: {}, avg_accuracy_reverse: {}, Avg_Hits H R T: {} {} {}, Avg_BCE_Acc H R T: {} {} {}, Avg_POS_BCE_Acc H R T: {} {} {}, Avg_NEG_BCE_Acc H R T: {} {} {}, Avg_Link_Prediction_Loss: {}, Time: {}'.format(
				epc, avg_loss_triple_classification, avg_klloss, avg_accuracy, avg_accuracy_reverse, avg_hits1['h'], avg_hits1['r'], avg_hits1['t'], avg_bce_acc['h'], avg_bce_acc['r'], avg_bce_acc['t'], avg_bce_acc_pos['h'], avg_bce_acc_pos['r'], avg_bce_acc_pos['t'], avg_bce_acc_neg['h'], avg_bce_acc_neg['r'], avg_bce_acc_neg['t'], avg_loss_link_prediction, time_epoch))

			if hyperparams['task'] == 'LP':
				self.link_prediction(epc)
			else:
				self.triple_classification(epc)
		if hyperparams['wandb'] : wandb.finish()

	def triple_classification(self, epc=-1, split='valid'):
		model = self.model
		tokenizer = self.tokenizer

		device = self.device
		hyperparams = self.hyperparams

		batch_size = 64
		neg_rate = 1

		data_loader = self.data_loader
		ent2id = data_loader.ent2id
		rel2id = data_loader.rel2id

		criterion = torch.nn.CrossEntropyLoss()

		model.eval()
		
		degrees = data_loader.statistics['degrees']
		
		with torch.no_grad():
			total_loss_triple_classification = 0

			total_accuracy = 0

			time_begin = time.time()
			
			if split == 'valid':
				data_sampler = data_loader.valid_data_sampler(batch_size=batch_size, neg_rate=neg_rate)
			else:
				data_sampler = data_loader.test_data_sampler(batch_size=batch_size, neg_rate=neg_rate)
			
			n_batch = len(data_sampler)
			dataset_size = data_sampler.get_dataset_size()


			for i_b, batch in tqdm(enumerate(data_sampler), total=n_batch):
				triples = [i[0] for i in batch]
				triple_degrees = [ [ degrees.get(e, 0)  for e in triple] for triple in triples]

				batch_size_ = len(batch)

				for mode in ['triple_classification']:
					inputs, positions = data_loader.batch_tokenize(triples, mode)
					inputs.to(device)
					
					if mode == 'triple_classification':
						labels = [i[1] for i in batch]
						labels = torch.tensor(labels).to(device)

						preds = model(inputs, positions, mode, triple_degrees)
						loss = criterion(preds, labels) 

						pred_labels = preds.argmax(dim=1)
						total_accuracy += (pred_labels == labels).int().sum().item()

						total_loss_triple_classification += loss.item() * batch_size_


						
					
			avg_loss_triple_classification = total_loss_triple_classification / dataset_size
			avg_accuracy = total_accuracy / dataset_size



			time_end = time.time()
			time_epoch = time_end - time_begin

			print('{} Triple Classification: Epoch: {} , Avg_Loss: {}, Avg_Accuracy: {}, Time: {}'.format(
				split, epc, avg_loss_triple_classification, avg_accuracy, time_epoch))

			if hyperparams['wandb']: wandb.log({'acc': avg_accuracy})

			if split != 'test':
				self.save_model(epc, 'acc', avg_accuracy)


		model.train()

	def link_prediction(self, epc=-1, split='valid'):
		model = self.model
		device = self.device
		hyperparams = self.hyperparams
		data_loader = self.data_loader

		n_ent = model.n_ent
		n_rel = model.n_rel

		ent2id = data_loader.ent2id
		rel2id = data_loader.rel2id
		entity_list = data_loader.entity_list

		model.eval()

		sigmoid = torch.nn.Sigmoid()


		dataset = data_loader.get_dataset(split)
		groundtruth = data_loader.get_groundtruth()

		dl_statistics = data_loader.statistics 
		max_degree = dl_statistics['max_degree']
		degree_group = dl_statistics['degree_group']
		count_degree_group = dl_statistics['count_degree_group']
		degrees = dl_statistics['degrees']


		ks = [1, 3, 10]
		MR = { 
				setting:
					{target: 0 for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			 } 

		MRR = { 
				setting:
					{target: 0 for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			 }

		hits = { 
				setting:
					{target: {k: 0 for k in ks} for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			 } 

		groups = [ 'given_{}'.format(i) for i in count_degree_group.keys()] + [ 'target_{}'.format(i) for i in count_degree_group.keys()] + [ 'both_{}'.format(i) for i in count_degree_group.keys()]
		MR_by_degree = { degree_group: { 
				setting:
					{target: 0 for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			 } for degree_group in (groups) } 
		
		MRR_by_degree = { degree_group: { 
				setting:
					{target: 0 for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			 } for degree_group in (groups) }

		hits_by_degree = { degree_group: { 
				setting:
					{target: {k: 0 for k in ks} for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			 } for degree_group in (groups) } 

		count_by_degree = { degree_group: { 
				setting:
					{target: 0 for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			 } for degree_group in (groups) }

		if hyperparams['contrastive']:
			# calc target embeddings 
			batch_size = 128
			ent_target_encoded = torch.zeros((n_ent, model.hidden_size)).to(device)
			rel_target_encoded = torch.zeros((n_rel, model.hidden_size)).to(device)

			
			with torch.no_grad():
				# calc entity target embeddings
				random_map = [ i for i in range(n_ent)]
				batch_list = [ random_map[i:i+batch_size] for i in range(0, n_ent, batch_size)] 
			
				for batch in batch_list:
					batch_targets = [ entity_list[_] for _ in batch]
					target_inputs, target_positions = data_loader.batch_tokenize_target(targets=batch_targets)
					target_inputs.to(device)
					target_encodes = model.encode_target(target_inputs, target_positions, mode='link_prediction_h')
					ent_target_encoded[batch] = target_encodes
		
		f = open(self.result_log, 'a')
		print('Load Path: {} split: {}'.format(self.load_path, split), file=f)

		with torch.no_grad():
			for target in ['head', 'tail']:
				count_triples = 0
				total_triples = len(dataset)
				candidate_degrees = [ degrees.get(tar, 0) for tar in entity_list]

				for given in tqdm(groundtruth[split][target].keys()):
					
					expects = groundtruth[split][target][given] 
					corrects = groundtruth['all'][target][given] 
					
					giv_rel = given[0]
					giv_ent = given[1]

					given_degree_group = degree_group.get(giv_ent, 0)

					if target == 'head':
						triples = [(expects[0], giv_rel, giv_ent)]
						mode = "link_prediction_h"

					else:
						triples = [(giv_ent, giv_rel, expects[0])]
						mode = "link_prediction_t"

					triple_degrees = [ [ degrees.get(e, 0) for e in triple] for triple in triples]
					ent_list_degrees = [ degrees.get(e, 0) for e in entity_list]


					inputs, positions = data_loader.batch_tokenize(triples, mode)
					inputs.to(device)	

					if not hyperparams['no_use_lm']:
						if hyperparams['contrastive']:
							target_preds = model(inputs, positions, mode)
							target_encodes = ent_target_encoded
						
							preds = model.match(target_preds, target_encodes, triple_degrees, mode, test=True, ent_list_degrees = ent_list_degrees)
						else:
							preds, confidence = model(inputs, positions, mode)
						preds = sigmoid(preds)
					
					if hyperparams['use_structure']:
						triple_score = model.forward_transe(positions, mode).squeeze()

						preds_transe = (margin - triple_score).sigmoid().unsqueeze(0)
						
						if hyperparams['no_use_lm']:
							preds = preds_transe
						
					scores = preds.squeeze() 

					tops = scores.argsort(descending=True).tolist()
					
					for expect in expects:
						target_degree_group = degree_group.get(expect, 0)
						
						expect_id = ent2id[expect]

						for setting in ['raw', 'filter']:
							if setting == 'raw':
								tops_ = tops 
								rank = tops_.index(expect_id) + 1 							
							else:
								other_corrects = [correct for correct in corrects if correct != expect]
								other_correct_ids = set([ent2id[c] for c in other_corrects])
								tops_ = [ t for t in tops if (not t in other_correct_ids)]

								rank = tops_.index(expect_id) + 1 


							MRR[setting][target] += 1/rank 
							MR[setting][target] += rank 

							MRR_by_degree['given_{}'.format(given_degree_group)][setting][target] += 1/rank 
							MR_by_degree['given_{}'.format(given_degree_group)][setting][target] += rank

							MRR_by_degree['target_{}'.format(target_degree_group)][setting][target] += 1/rank 
							MR_by_degree['target_{}'.format(target_degree_group)][setting][target] += rank 

							MRR_by_degree['both_{}'.format(given_degree_group)][setting][target] += 1/rank 
							MR_by_degree['both_{}'.format(given_degree_group)][setting][target] += rank 
							MRR_by_degree['both_{}'.format(target_degree_group)][setting][target] += 1/rank 
							MR_by_degree['both_{}'.format(target_degree_group)][setting][target] += rank 

							count_by_degree['given_{}'.format(given_degree_group)][setting][target] += 1
							count_by_degree['target_{}'.format(target_degree_group)][setting][target] += 1 

							count_by_degree['both_{}'.format(given_degree_group)][setting][target] += 1
							count_by_degree['both_{}'.format(target_degree_group)][setting][target] += 1 
		
							for k in ks:
								if rank <= k:
									hits[setting][target][k] += 1
									hits_by_degree['given_{}'.format(given_degree_group)][setting][target][k] += 1
									hits_by_degree['target_{}'.format(target_degree_group)][setting][target][k] += 1 

									hits_by_degree['both_{}'.format(given_degree_group)][setting][target][k] += 1
									hits_by_degree['both_{}'.format(target_degree_group)][setting][target][k] += 1 



						count_triples += 1


				
				for setting in ['raw', 'filter']:
					MR[setting][target] /= count_triples
					MRR[setting][target] /= count_triples
					for k in ks:
						hits[setting][target][k] /= count_triples

					print('MR {0:.5f} MRR {1:.5f} hits 1 {2:.5f} 3 {3:.5f} 10 {4:.5f}, Setting: {5} Target: {6} '.format(
						MR[setting][target], MRR[setting][target], hits[setting][target][1], hits[setting][target][3], hits[setting][target][10],
						setting, target
					 ))

					print('MR {0:.5f} MRR {1:.5f} hits 1 {2:.5f} 3 {3:.5f} 10 {4:.5f}, Setting: {5} Target: {6} '.format(
						MR[setting][target], MRR[setting][target], hits[setting][target][1], hits[setting][target][3], hits[setting][target][10],
						setting, target
					 ), file=f)


		if split == 'test' or epc % 10 == 0:
			for setting in ['filter']:
				for group in groups:
					# Average over predicting head or tail
					if not group.startswith('both'):

						mr_head = ((MR_by_degree[group][setting]['head'] / max(count_by_degree[group][setting]['head'], 1))) 
						mrr_head = ((MRR_by_degree[group][setting]['head'] / max(count_by_degree[group][setting]['head'], 1))) 
						hits1_head = ((hits_by_degree[group][setting]['head'][1] / max(count_by_degree[group][setting]['head'], 1))) 
						hits3_head = ((hits_by_degree[group][setting]['head'][3] / max(count_by_degree[group][setting]['head'], 1))) 
						hits10_head = ((hits_by_degree[group][setting]['head'][10] / max(count_by_degree[group][setting]['head'], 1))) 

						mr_tail = ((MR_by_degree[group][setting]['tail'] / max(count_by_degree[group][setting]['tail'], 1))) 
						mrr_tail = ((MRR_by_degree[group][setting]['tail'] / max(count_by_degree[group][setting]['tail'], 1))) 
						hits1_tail = ((hits_by_degree[group][setting]['tail'][1] / max(count_by_degree[group][setting]['tail'], 1))) 
						hits3_tail = ((hits_by_degree[group][setting]['tail'][3] / max(count_by_degree[group][setting]['tail'], 1))) 
						hits10_tail = ((hits_by_degree[group][setting]['tail'][10] / max(count_by_degree[group][setting]['tail'], 1))) 

						mr = int((mr_head + mr_tail) / 2 * 1000) / 1000
						mrr = int((mrr_head + mrr_tail) / 2 * 1000) / 1000
						hits1 = int((hits1_head + hits1_tail) / 2 * 1000) / 1000
						hits3 = int((hits3_head + hits3_tail) / 2 * 1000) / 1000
						hits10 = int((hits10_head + hits10_tail) / 2 * 1000) / 1000

						print('Distinguish Given / Target: Group: {} mr {} mrr {} hits1 {} hits3 {} hits10 {}'.format(group, mr, mrr, hits1, hits3, hits10))

						print('Distinguish Given / Target: Group: {} mr {} mrr {} hits1 {} hits3 {} hits10 {}'.format(group, mr, mrr, hits1, hits3, hits10), file=f)

					else:
						mr = int((((MR_by_degree[group][setting]['head'] + MR_by_degree[group][setting]['tail']) / max(count_by_degree[group][setting]['head'] + count_by_degree[group][setting]['tail'], 1))) * 1000) / 1000
						mrr = int((((MRR_by_degree[group][setting]['head'] + MRR_by_degree[group][setting]['tail']) / max(count_by_degree[group][setting]['head'] + count_by_degree[group][setting]['tail'], 1))) * 1000) / 1000
						hits1 = int((((hits_by_degree[group][setting]['head'][1] + hits_by_degree[group][setting]['tail'][1]) / max(count_by_degree[group][setting]['head'] + count_by_degree[group][setting]['tail'], 1))) * 1000) / 1000
						hits3 = int((((hits_by_degree[group][setting]['head'][3] + hits_by_degree[group][setting]['tail'][3]) / max(count_by_degree[group][setting]['head'] + count_by_degree[group][setting]['tail'], 1))) * 1000) / 1000
						hits10 = int((((hits_by_degree[group][setting]['head'][10] + hits_by_degree[group][setting]['tail'][10]) / max(count_by_degree[group][setting]['head'] + count_by_degree[group][setting]['tail'], 1))) * 1000) / 1000

						total = (count_by_degree[group][setting]['head'] + count_by_degree[group][setting]['tail']) / 2
						print('General: Group: {} mr {} mrr {} hits1 {} hits3 {} hits10 {} total {}'.format(group, mr, mrr, hits1, hits3, hits10, total))

					


		raw_mrr = (MRR['raw']['head'] + MRR['raw']['tail']) / 2
		raw_hits1 = (hits['raw']['head'][1] + hits['raw']['tail'][1]) / 2
		raw_hits3 = (hits['raw']['head'][3] + hits['raw']['tail'][3]) / 2
		raw_hits10 = (hits['raw']['head'][10] + hits['raw']['tail'][10]) / 2


		fil_mr = (MR['filter']['head'] + MR['filter']['tail']) / 2
		fil_mrr = (MRR['filter']['head'] + MRR['filter']['tail']) / 2
		fil_hits1 = (hits['filter']['head'][1] + hits['filter']['tail'][1]) / 2
		fil_hits3 = (hits['filter']['head'][3] + hits['filter']['tail'][3]) / 2
		fil_hits10 = (hits['filter']['head'][10] + hits['filter']['tail'][10]) / 2

		print('Overall: MR {0:.5f} MRR {1:.5f} hits 1 {2:.5f} 3 {3:.5f} 10 {4:.5f}, Setting: Filter '.format(
			fil_mr, fil_mrr, fil_hits1, fil_hits3, fil_hits10), file=f)
		print('Overall: MR {0:.5f} MRR {1:.5f} hits 1 {2:.5f} 3 {3:.5f} 10 {4:.5f}, Setting: Filter '.format(
			fil_mr, fil_mrr, fil_hits1, fil_hits3, fil_hits10))
		f.close()

		if split != 'test':
			self.update_metric(epc, 'raw_mrr', raw_mrr)
			self.update_metric(epc, 'raw_hits1', raw_hits1)
			self.update_metric(epc, 'raw_hits3', raw_hits3)
			self.update_metric(epc, 'raw_hits10', raw_hits10)

			self.update_metric(epc, 'fil_mr', fil_mr)
			#self.update_metric(epc, 'fil_mrr', fil_mrr)
			self.save_model(epc, 'fil_mrr', fil_mrr)
			#self.update_metric(epc, 'fil_hits1', fil_hits1)
			self.save_model(epc, 'fil_hits1', fil_hits1)
			self.update_metric(epc, 'fil_hits3', fil_hits3)
			#self.update_metric(epc, 'fil_hits10', fil_hits10)
			self.save_model(epc, 'fil_hits10', fil_hits10)

			if hyperparams['wandb']: wandb.log({'fil_mr': fil_mr, 'fil_mrr': fil_mrr, 'fil_hits1': fil_hits1, 'fil_hits3': fil_hits3, 'fil_hits10': fil_hits10, 'raw_hits10': raw_hits10})

		model.train()


	def update_metric(self, epc, name, score):
		self.history_value[name].append(score)
		if ( name not in ['fil_mr', 'raw_mr'] and score > self.best_metric[name]) or ( name in ['fil_mr', 'raw_mr'] and score < self.best_metric[name]):
			self.best_metric[name] = score
			self.best_epoch[name] = epc
			if name in ['fil_mr', 'raw_mr']:
				print('! Metric {0} Updated as: {1:.2f}'.format(name, score))
			else:
				print('! Metric {0} Updated as: {1:.2f}'.format(name, score*100))
			return True
		else:
			return False

	def save_model(self, epc, metric, metric_val):
		save_path = self.param_path_template.format(epc, metric)
		last_path = self.param_path_template.format(self.best_epoch[metric], metric)

		if self.update_metric(epc, metric, metric_val):
			if os.path.exists(last_path) and save_path != last_path and epc >= self.best_epoch[metric]:
				os.remove(last_path)
				print('Last parameters {} deleted'.format(last_path))
			
			#torch.save(self.model.state_dict(), save_path)
			torch.save({
				'model_state_dict': self.model.state_dict(), 
				'optimizer_state_dict': self.optimizer.state_dict(),
				'scheduler_state_dict': self.scheduler.state_dict(),
			}, save_path)

			print('Parameters saved into ', save_path)


	def debug_signal_handler(self, signal, frame):
		pdb.set_trace()
	
	def log_best(self):
		print('Best Epoch {0} micro_f1 {1}'.format(self.best_epoch, self.best_metric))
			