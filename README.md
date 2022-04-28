# Language Models as Knowledge Embeddings

Source code for the paper Language Models as Knowledge Embeddings

## Requirements

- [PyTorch](http://pytorch.org/) version >= 1.7.1
- [NumPy](http://numpy.org/) version >= 1.19.5
- transformers
- tqdm
- Python version >= 3.6

## Usage

Run main.py to train or test our models. 

An example for training for triple classification:

```bash
python main.py --batch_size 16 --plm bert --data wn18rr --task TC
```

An example for training for link prediction:

```bash
python main.py --batch_size 16 --plm bert --contrastive --self_adversarial --data wn18rr --task LP 
```

The arguments are as following:
* `--bert_lr`: learning rate of the language model.
* `--model_lr`: learning rate of other parameters.
* `--batch_size`: batch size used in training.
* `--weight_decay`: weight dacay used in training.
* `--data`: name of the dataset. Choose from 'fb15k-237', 'wn18rr', 'fb13' and 'umls'.
* `--plm`: choice of the language model. Choose from 'bert' and 'bert_tiny'.
* `--load_path`: path of checkpoint to load.
* `--load_epoch`: load the checkpoint of a specific epoch. Use with --load_metric.
* `--load_metric`: use with --load_epoch.
* `--link_prediction`: run link prediction evaluation after loading a checkpoint.
* `--triple_classification`: run triple classification evaluation after loading a checkpoint.
* `--self_adversarial`: use self-adversarial negative sampling for efficient KE learning.
* `--contrastive`: use contrastive LMKE.
* `--task`: specify the task. Choose from 'LP' (link prediction) and 'TC' (triple classification).


### Datasets

The datasets are put in the folder 'data', including fb15k-237, WN18RR, FB13 and umls.


