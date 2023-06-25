# Language Models as Knowledge Embeddings

Source code for the paper Language Models as Knowledge Embeddings

## Notice

[June 2023] We recently identified a data leakage issue in our code that, during prediction, we inadvertently leaked degree information about the entities to be predicted. This unintentionally provided a shortcut for the model, which affected the experimental results to some extent. We have fixed this issue and re-conducted our experiments, and updated the paper accordingly. The revised results do not impact the majority of the paper's conclusions and contributions. The method continues to achieve state-of-the-art (SOTA) performance on the WN18RR, FB13, and WN11 datasets, compared to previous works. However, on the FB15k-237 dataset, the model's performance has declined to a certain extent and underperforms state-of-the-art structured-based methods. We sincerely apologize for this error.

Updated Results:

### WN18RR
| Methods | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|--------------------------------------|----|-----|--------|--------|---------|
| TransE                               | 2300 | 0.243 | 0.043 | 0.441 | 0.532 |
| DistMult                             | 5110 | 0.430 | 0.390 | 0.440 | 0.490 |
| ComplEx                              | 5261 | 0.440 | 0.410 | 0.460 | 0.510 |
| RotatE                               | 3340 | 0.476 | 0.428 | 0.492 | 0.571 |
| TuckER                               | - | 0.470 | 0.443 | 0.482 | 0.526 |
| HAKE                                 | - | 0.497 | 0.452 | 0.516 | 0.582 |
| CoKE                                 | - | 0.484 | 0.450 | 0.496 | 0.553 |
|----------------------------------------|----|-----|--------|--------|---------|
| Pretrain-KGE_TransE                   | 1747 | 0.235 | - | - | 0.557 |
| KG-BERT                               | 97 | 0.216 | 0.041 | 0.302 | 0.524 |
| StAR_BERT-base                        | 99 | 0.364 | 0.222 | 0.436 | 0.647 |
| MEM-KGC_BERT-base_(w/o EP)            | - | 0.533 | 0.473 | 0.570 | 0.636 |
| MEM-KGC_BERT-base_(w/ EP)             | - | 0.557 | 0.475 | 0.604 | 0.704 |
| C-LMKE_BERT-base                      | 79 | 0.619 | 0.523 | 0.671 | 0.789 |
 

### FB15k-237

| Methods | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|--------------------------------------|----|-----|--------|--------|---------|
| TransE                               | 323 | 0.279 | 0.198 | 0.376 | 0.441 |
| DistMult                             | 254 | 0.241 | 0.155 | 0.263 | 0.419 |
| ComplEx                              | 339 | 0.247 | 0.158 | 0.275 | 0.428 |
| RotatE                               | 177 | 0.338 | 0.241 | 0.375 | 0.533 |
| TuckER                               | - | 0.358 | 0.266 | 0.394 | 0.544 |
| HAKE                                 | - | 0.346 | 0.250 | 0.381 | 0.542 |
| CoKE                                 | - | 0.364 | 0.272 | 0.400 | 0.549 |
|----------------------------------------|----|-----|--------|--------|---------|
| Pretrain-KGE_TransE                   | 162 | 0.332 | - | - | 0.529 |
| KG-BERT                               | 153 | - | - | - | 0.420 |
| StAR_BERT-base                        | 136 | 0.263 | 0.171 | 0.287 | 0.452 |
| MEM-KGC_BERT-base_(w/o EP)            | - | 0.339 | 0.249 | 0.372 | 0.522 |
| MEM-KGC_BERT-base_(w/ EP)             | - | 0.346 | 0.253 | 0.381 | 0.531 |
| C-LMKE_BERT-base                      | 141 | 0.306 | 0.218 | 0.331 | 0.484 |


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


