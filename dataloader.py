import os
#import pdb
import random
import math
import pickle
import torch
import time 
from tqdm import tqdm
import copy
from transformers import BatchEncoding

count_sampler = 0
num_limit= 100

class DataSampler(object):
	def __init__(self, datasetName, mode, pos_dataset, whole_dataset, batch_size, entity_set, relation_set, neg_rate, groundtruth=None, possible_entities=None, rdrop=False, pos_neg_dataset=None):
		self.datasetName = datasetName

		self.batch_size = batch_size
		self.entity_set = entity_set
		self.relation_set = relation_set

		self.mode = mode
		self.whole_dataset = whole_dataset
		self.pos_neg_dataset = pos_neg_dataset # dataset with originally provided negative samples 

		self.neg_rate = neg_rate
		self.groundtruth = groundtruth
		self.possible_entities = possible_entities

		self.rdrop = rdrop

		if not os.path.exists('./sampler'):
			os.makedirs('./sampler')

		global count_sampler
		count_sampler += 1
		dataset_path = 'sampler/{}-{}-{}-{}.pkl'.format(datasetName, mode, neg_rate, count_sampler)

		if self.datasetName in ['fb13'] and mode != 'train':
			self.dataset = [ ((i[0], i[1], i[2]), i[3]) for i in self.pos_neg_dataset]

		else:
			if os.path.exists(dataset_path):
				with open(dataset_path, 'rb') as fil:
					self.dataset = pickle.load(fil)
			else:
				self.dataset = self.create_dataset(pos_dataset)
				with open(dataset_path, 'wb') as fil:
					pickle.dump(self.dataset, fil)

		
		self.n_batch = math.ceil(len(self.dataset) / batch_size)
		self.i_batch = 0

		assert(batch_size % (1+neg_rate) == 0)

	def create_dataset(self, pos_dataset):
		dataset = [] 
		random.shuffle(pos_dataset)
		pos_dataset_set = set(pos_dataset)
		whole_dataset_set = set(self.whole_dataset)

		if self.mode == 'train':
			random_ratio = 1#1/3
			constrain_ratio = 0#1/3
			reverse_ratio = 0#1/3
			viewable = 'train'
			viewable_set = pos_dataset_set
		else:
			random_ratio = 1
			constrain_ratio = 0
			reverse_ratio = 0
			viewable = 'all'
			viewable_set = whole_dataset_set


		for triple in tqdm(pos_dataset):
			dataset.append((triple, 1))
			choice = random_choose(random_ratio, constrain_ratio, reverse_ratio)

			h, r, t = triple 
			for _ in range(self.neg_rate):
				count = 0
				while (True):
					if (random.sample(range(2), 1)[0] == 1):
						# replace head
						if choice == 'random':
							candidate_ents = self.entity_set - set(self.groundtruth[viewable]['head'][(r, t)])
							replace_ent = random.sample(candidate_ents, 1)[0]
							neg_triple = (replace_ent, r, t)
						elif choice == 'constrain':
							candidate_ents = self.possible_entities['train']['head'][r] - set(self.groundtruth[viewable]['head'][(r, t)])
							# head that are head of rel, but not head of (rel, tail) 
							if len(candidate_ents) == 0:
								candidate_ents = self.entity_set - set(self.groundtruth[viewable]['head'][(r, t)])
							replace_ent = random.sample(candidate_ents, 1)[0]
							neg_triple = (replace_ent, r, t)
						else: # choice == 'reverse'
							neg_triple = (t, r, h)
					else:
						# replace tail
						if choice == 'random':
							candidate_ents = self.entity_set - set(self.groundtruth[viewable]['tail'][(r, h)])
							replace_ent = random.sample(candidate_ents, 1)[0]
							neg_triple = (h, r, replace_ent)
						elif choice == 'constrain':
							candidate_ents = self.possible_entities['train']['tail'][r] - set(self.groundtruth[viewable]['tail'][(r, h)])
							# tail that are tail of rel, but not tail of (rel, head) 
							if len(candidate_ents) == 0:
								candidate_ents = self.entity_set - set(self.groundtruth[viewable]['tail'][(r, h)])
							replace_ent = random.sample(candidate_ents, 1)[0]
							neg_triple = (h, r, replace_ent)
						else: # choice == 'reverse':
							neg_triple = (t, r, h)

					if neg_triple not in viewable_set:
						dataset.append((neg_triple, 0))
						break 
					elif choice == 'reverse':
						dataset.append((neg_triple, 1))


		return dataset


	def __iter__(self):
		return self 

	def __next__(self):
		if self.i_batch == self.n_batch:
			raise StopIteration()

		batch = self.dataset[self.i_batch*self.batch_size: (self.i_batch+1)*self.batch_size]
		if self.rdrop:
			batch = batch + batch 
			
		self.i_batch += 1
		return batch

	def __len__(self):
		return self.n_batch

	def get_dataset_size(self):
		return len(self.dataset)


class DataLoader(object):
	def __init__(self, in_paths, tokenizer, batch_size = 16, neg_rate = 1, add_tokens=False, p_tuning=False, rdrop=False, model='bert'):
		
		self.datasetName = in_paths['dataset'] 

		self.train_set = self.load_dataset(in_paths['train'])
		if self.datasetName not in ['fb13']:
			self.valid_set = self.load_dataset(in_paths['valid'])
			self.test_set = self.load_dataset(in_paths['test'])
			self.valid_set_with_neg = None
			self.test_set_with_neg = None
		else:
			self.valid_set, self.valid_set_with_neg = self.load_dataset_with_neg(in_paths['valid'])
			self.test_set, self.test_set_with_neg = self.load_dataset_with_neg(in_paths['test'])


		self.whole_set = self.train_set + self.valid_set + self.test_set

		self.uid2text =  {}
		self.uid2tokens =  {}

		self.entity_set = set([t[0] for t in (self.train_set + self.valid_set + self.test_set)] + [t[-1] for t in (self.train_set + self.valid_set + self.test_set)])
		self.relation_set = set([t[1] for t in (self.train_set + self.valid_set + self.test_set)])

		self.tokenizer = tokenizer
		for p in in_paths['text']:
			self.load_text(p)

		self.batch_size = batch_size
		self.step_per_epc = math.ceil(len(self.train_set) * (1+neg_rate) / batch_size)


		self.train_entity_set = set([t[0] for t in self.train_set] + [t[-1] for t in self.train_set])
		self.train_relation_set = set([t[1] for t in self.train_set])


		self.entity_list = sorted(self.entity_set)
		self.relation_list = sorted(self.relation_set)

		self.ent2id = {e:i for i, e in enumerate(sorted(self.entity_set))}
		self.rel2id = {r:i for i, r in enumerate(sorted(self.relation_set))}

		self.id2ent = {i:e for i, e in enumerate(sorted(self.entity_set))}
		self.id2rel = {i:r for i, r in enumerate(sorted(self.relation_set))}
		

		self.neg_rate = neg_rate

		self.groundtruth, self.possible_entities= self.count_groundtruth()

		self.add_tokens = add_tokens
		self.p_tuning = p_tuning
		self.rdrop = rdrop

		self.model = model 

		self.orig_vocab_size = len(tokenizer)

		self.count_degrees()

		# few shot 
		#self.train_set = self.train_set[:num_limit]

		'''
		self.train_set = [i for i in self.train_set if self.statistics['degrees'][i[0]]  < 3 or self.statistics['degrees'][i[-1]]  < 3]
		self.valid_set = [i for i in self.valid_set if self.statistics['degrees'][i[0]]  < 3 or self.statistics['degrees'][i[-1]]  < 3]
		self.test_set = [i for i in self.test_set if self.statistics['degrees'][i[0]]  < 3 or self.statistics['degrees'][i[-1]]  < 3]

		print('Num Train {} Valid {} Test {}'.format(len(self.train_set), len(self.valid_set), len(self.test_set)))
		'''

	def load_dataset(self, in_path):
		dataset = []
		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				if in_path[-3:] == 'txt':
					h, t, r = line.strip('\n').split('\t')
				else:
					h, r, t = line.strip('\n').split('\t')
				dataset.append((h, r, t))
		return dataset 

	def load_dataset_with_neg(self, in_path):
		dataset = []
		dataset_with_neg = []
		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				h, r, t, l = line.strip('\n').split('\t')

				if l == '-1':
					l = 0
				else:
					l = 1
				dataset.append((h, r, t))
				dataset_with_neg.append((h, r, t, l))
		return dataset, dataset_with_neg

	def load_name_wiki(self, in_path):
		uid2name = {}
		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				_ = line.strip('\n').split('\t')
				uid = _[0]
				name = _[1]
				uid2name[uid] = name
		return uid2name
				

	def load_text(self, in_path):
		uid2text = self.uid2text
		uid2tokens = self.uid2tokens

		tokenizer = self.tokenizer


		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				uid, text = line.strip('\n').split('\t', 1)
				text = text.replace('@en', '').strip('"')
				if uid not in uid2text.keys():
					uid2text[uid] = text 

				tokens = tokenizer.tokenize(text)

				if uid not in uid2tokens.keys():
					uid2tokens[uid] = tokens


		self.uid2text = uid2text
		self.uid2tokens = uid2tokens

	def triple_to_text(self, triple, with_text):

		tokenizer = self.tokenizer
		ent2id = self.ent2id
		rel2id = self.rel2id

		if True:
			# 512 tokens, 1 CLS, 1 SEP, 1 head, 1 rel, 1 tail, so 507 remaining.
			h_n_tokens = 64 #241
			t_n_tokens = 64 #241
			r_n_tokens = 16

		h, r, t = triple
		
		h_text_tokens =  self.uid2tokens.get(h, [])[:h_n_tokens] if with_text['h'] else []		
		r_text_tokens =  self.uid2tokens.get(r, [])[:r_n_tokens] if with_text['r'] else []
		t_text_tokens =  self.uid2tokens.get(t, [])[:t_n_tokens] if with_text['t'] else []

		
		if self.add_tokens:
			if self.p_tuning:
				h_token = ["[head_b1]", "[head_b2]"] + (['[ent_{}]'.format(ent2id[h])] if with_text['h'] else [tokenizer.mask_token]) + ["[head_a1]", "[head_a2]"]
				r_token = ["[rel_b1]", "[rel_b2]"] + (['[rel_{}]'.format(rel2id[r])] if with_text['r'] else [tokenizer.mask_token]) + ["[rel_a1]", "[rel_a2]"]
				t_token = ["[tail_b1]", "[tail_b2]"] + (['[ent_{}]'.format(ent2id[t])] if with_text['t'] else [tokenizer.mask_token]) + ["[tail_a1]", "[tail_a2]"]
			else:
				h_token = ['[ent_{}]'.format(ent2id[h])] if with_text['h'] else [tokenizer.mask_token]
				r_token = ['[rel_{}]'.format(rel2id[r])] if with_text['r'] else [tokenizer.mask_token]
				t_token = ['[ent_{}]'.format(ent2id[t])] if with_text['t'] else [tokenizer.mask_token]
		else:
			h_token = ['[CLS]'] if with_text['h'] else [tokenizer.mask_token]
			r_token = ['[CLS]'] if with_text['r'] else [tokenizer.mask_token]
			t_token = ['[CLS]'] if with_text['t'] else [tokenizer.mask_token]


		tokens = h_token + h_text_tokens + r_token + r_text_tokens + t_token + t_text_tokens  
		
		text = tokenizer.convert_tokens_to_string(tokens)

		return text, tokens

	def element_to_text(self, target):
		tokenizer = self.tokenizer
		ent2id = self.ent2id
		rel2id = self.rel2id
		
		n_tokens = 125 #508

		text_tokens = self.uid2tokens.get(target, [])[:n_tokens] 
		
		if self.add_tokens:
			if target in ent2id.keys():
				token = ['[ent_{}]'.format(ent2id[target])]
			else:
				token = ['[rel_{}]'.format(rel2id[target])]
		else:
			token = ['[CLS]']

		tokens = token + text_tokens 
		
		text = tokenizer.convert_tokens_to_string(tokens)

		return text, tokens

	def batch_tokenize(self, batch_triples, mode):
		batch_texts = []
		batch_tokens = []
		batch_positions = []

		ent2id = self.ent2id
		rel2id = self.rel2id

		if mode in ['triple_classification']:
			with_text = {'h': True, 'r': True, 't': True}
		elif mode == "link_prediction_h":
			with_text = {'h': False, 'r': True, 't': True}
		elif mode == "link_prediction_r":
			with_text = {'h': True, 'r': False, 't': True}
		elif mode == "link_prediction_t":
			with_text = {'h': True, 'r': True, 't': False}


		for triple in batch_triples:
			text, tokens = self.triple_to_text(triple, with_text)
			batch_texts.append(text)
			batch_tokens.append(tokens)

			h, r, t = triple


		#batch_tokens_ = self.tokenizer(batch_texts, truncation = True, max_length = 512, return_tensors='pt', padding=True )
		batch_tokens = self.my_tokenize(batch_tokens, max_length=512, padding=True, model=self.model)


		orig_vocab_size = self.orig_vocab_size
		num_ent_rel_tokens = len(ent2id) + len(rel2id)

		mask_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
		cls_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)

		for i, _ in enumerate(batch_tokens['input_ids']):
			triple = batch_triples[i]
			h, r, t = triple

			if not self.add_tokens:
				cls_pos, h_pos, r_pos, t_pos = torch.where((_==mask_idx) + (_==cls_idx))[0]
			else:
				h_pos, r_pos, t_pos = torch.where( (_ >= orig_vocab_size) * (_ < orig_vocab_size + num_ent_rel_tokens) + (_ == mask_idx) )[0]


			batch_positions.append({'head': (ent2id[h], h_pos.item()), 'rel': (rel2id[r], r_pos.item()), 'tail': (ent2id[t], t_pos.item())})

		return batch_tokens, batch_positions


	def batch_tokenize_target(self, batch_triples=None, mode=None, targets = None):
		batch_texts = []
		batch_tokens = []
		batch_positions = []

		ent2id = self.ent2id
		rel2id = self.rel2id

		if targets == None:
			if mode == "link_prediction_h":
				targets = [ triple[0] for triple in batch_triples]
			elif mode == "link_prediction_r":
				targets = [ triple[1] for triple in batch_triples]
			elif mode == "link_prediction_t":
				targets = [ triple[2] for triple in batch_triples]

		for target in targets:
			text, tokens = self.element_to_text(target)
			batch_texts.append(text)
			batch_tokens.append(tokens)
		
		batch_tokens = self.my_tokenize(batch_tokens, max_length=512, padding=True, model=self.model)

		for i, _ in enumerate(batch_tokens['input_ids']):
			target = targets[i]
			target_pos = 1
			
			if target in ent2id.keys():
				target_idx = ent2id[target]
			else:
				target_idx = rel2id[target]

			batch_positions.append( (target_idx, target_pos) )

		return batch_tokens, batch_positions

	def train_data_sampler(self):
		return DataSampler(datasetName = self.datasetName, mode='train', pos_dataset=self.train_set, whole_dataset=self.whole_set, batch_size=self.batch_size, 
			entity_set=self.train_entity_set, relation_set=self.train_relation_set, neg_rate=self.neg_rate, groundtruth=self.groundtruth, 
			possible_entities=self.possible_entities, rdrop=self.rdrop)

	def valid_data_sampler(self, batch_size, neg_rate):
		return DataSampler(datasetName = self.datasetName, mode='valid', pos_dataset=self.valid_set, whole_dataset=self.whole_set, batch_size=batch_size, 
			entity_set=self.entity_set, relation_set=self.relation_set, neg_rate=neg_rate, groundtruth=self.groundtruth, pos_neg_dataset=self.valid_set_with_neg)

	def test_data_sampler(self, batch_size, neg_rate):
		return DataSampler(datasetName = self.datasetName, mode='test', pos_dataset=self.test_set, whole_dataset=self.whole_set, batch_size=batch_size, 
			entity_set=self.entity_set, relation_set=self.relation_set, neg_rate=neg_rate, groundtruth=self.groundtruth, pos_neg_dataset=self.test_set_with_neg)

	def get_dataset_size(self, split='train'):
		if split == 'train':
			return len(self.train_set) * (1+self.neg_rate)

	def count_groundtruth(self):
		groundtruth = { split: {'head': {}, 'rel': {}, 'tail': {}} for split in ['all', 'train', 'valid', 'test']}
		possible_entities = { split: {'head': {}, 'tail': {}} for split in ['train']}

		for triple in self.train_set:
			h, r, t = triple
			groundtruth['all']['head'].setdefault((r, t), [])
			groundtruth['all']['head'][(r, t)].append(h)
			groundtruth['all']['tail'].setdefault((r, h), [])
			groundtruth['all']['tail'][(r, h)].append(t)
			groundtruth['all']['rel'].setdefault((h, t), [])
			groundtruth['all']['rel'][(h, t)].append(r)  
			groundtruth['train']['head'].setdefault((r, t), [])
			groundtruth['train']['head'][(r, t)].append(h)
			groundtruth['train']['tail'].setdefault((r, h), [])
			groundtruth['train']['tail'][(r, h)].append(t) 
			groundtruth['train']['rel'].setdefault((h, t), [])
			groundtruth['train']['rel'][(h, t)].append(r) 
			possible_entities['train']['head'].setdefault(r, set())
			possible_entities['train']['head'][r].add(h)
			possible_entities['train']['tail'].setdefault(r, set())
			possible_entities['train']['tail'][r].add(t)
		

		for triple in self.valid_set:
			h, r, t = triple
			groundtruth['all']['head'].setdefault((r, t), [])
			groundtruth['all']['head'][(r, t)].append(h)
			groundtruth['all']['tail'].setdefault((r, h), [])
			groundtruth['all']['tail'][(r, h)].append(t)
			groundtruth['all']['rel'].setdefault((h, t), [])
			groundtruth['all']['rel'][(h, t)].append(r)   
			groundtruth['valid']['head'].setdefault((r, t), [])
			groundtruth['valid']['head'][(r, t)].append(h)
			groundtruth['valid']['tail'].setdefault((r, h), [])
			groundtruth['valid']['tail'][(r, h)].append(t) 

		for triple in self.test_set:
			h, r, t = triple


			groundtruth['all']['head'].setdefault((r, t), [])
			groundtruth['all']['head'][(r, t)].append(h)
			groundtruth['all']['tail'].setdefault((r, h), [])
			groundtruth['all']['tail'][(r, h)].append(t)
			groundtruth['all']['rel'].setdefault((h, t), [])
			groundtruth['all']['rel'][(h, t)].append(r)   
			groundtruth['test']['head'].setdefault((r, t), [])
			groundtruth['test']['head'][(r, t)].append(h)
			groundtruth['test']['tail'].setdefault((r, h), [])
			groundtruth['test']['tail'][(r, h)].append(t) 


		return groundtruth, possible_entities


	def get_groundtruth(self):
		return self.groundtruth

	def get_dataset(self, split):
		assert (split in ['train', 'valid', 'test'])
		
		if split == 'train':
			return self.train_set
		elif split == 'valid':
			return self.valid_set
		elif split == 'test':
			return self.test_set

	def my_tokenize(self, batch_tokens, max_length=512, padding=True, model='roberta'):
		if model == 'roberta':
			start_tokens = ['<s>']
			end_tokens = ['</s>']
			pad_token = '<pad>'
		elif model == 'bert':
			start_tokens = ['[CLS]']
			end_tokens = ['[SEP]']
			pad_token = '[PAD]'

		batch_tokens = [ start_tokens + i + end_tokens for i in batch_tokens] 

		batch_size = len(batch_tokens)
		longest = min(max([len(i) for i in batch_tokens]), 512)

		if model == 'bert':
			input_ids = torch.zeros((batch_size, longest)).long()
		elif model == 'roberta':
			input_ids = torch.ones((batch_size, longest)).long()

		token_type_ids = torch.zeros((batch_size, longest)).long()
		attention_mask = torch.zeros((batch_size, longest)).long()

		for i in range(batch_size):
			tokens = self.tokenizer.convert_tokens_to_ids(batch_tokens[i])
			input_ids[i, :len(tokens)] = torch.tensor(tokens).long() 
			attention_mask[i, :len(tokens)] = 1

		if model == 'roberta':
			return BatchEncoding(data = {'input_ids': input_ids, 'attention_mask': attention_mask})
		elif model == 'bert':
			return BatchEncoding(data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})


	def adding_tokens(self):
		n_ent = len(self.ent2id)
		n_rel = len(self.rel2id)

		new_tokens = ["[ent_{}]".format(i) for i in range(n_ent)] + ["[rel_{}]".format(i) for i in range(n_rel)]
		
		if self.p_tuning:
			new_tokens += ["[head_b1]", "[head_b2]", "[head_a1]", "[head_a2]", 
							"[rel_b1]", "[rel_b2]", "[rel_a1]", "[rel_a2]", 
							"[tail_b1]", "[tail_b2]", "[tail_a1]", "[tail_a2]"] # continuous prompt 
	
		self.tokenizer.add_tokens(new_tokens)

	def count_degrees(self):
		train_set = self.train_set #+ self.valid_set + self.test_set
		degrees = {}

		
		for triple in train_set:
			h, r, t=triple
			degrees[h] = degrees.get(h, 0) + 1
			degrees[t] = degrees.get(t, 0) + 1
			degrees[r] = degrees.get(r, 0) + 1

		raw_degrees = copy.deepcopy(degrees)

		max_degree = 0
		for k, v in degrees.items():
			max_degree = max(max_degree, v)
		max_degree = math.floor(math.log(max_degree) / math.log(2))
		count_degree_group = { i:0 for i in range(0, max_degree+1)}

		for k, v in degrees.items():
			degrees[k] = math.floor(math.log(v) / math.log(2)) + 1
			count_degree_group[degrees[k]] = count_degree_group.get(degrees[k], 0) + 1

		self.statistics = {
			'degrees': raw_degrees,
			'degree_group': degrees,
			'count_degree_group': count_degree_group,
			'max_degree': max_degree
		}


def random_choose(random_ratio, constrain_ratio, reverse_ratio):
	x = random.random()
	if x <= random_ratio:
		return 'random'
	elif x <= (random_ratio + constrain_ratio):
		return 'constrain'
	else:
		return 'reverse'
