# Pytorch Recurrent Variational Autoencoder 

## Model:
This is the implementation of Samuel Bowman's [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349#)
with Kim's [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615) embedding for tokens

## Sampling examples:
> the new machine could be used to increase the number of ventures block in the company 's \<unk> shopping system to finance diversified organizations

> u.s. government officials also said they would be willing to consider whether the proposal could be used as urging and programs

> men believe they had to go on the \<unk> because their \<unk> were \<unk> expensive important

> the companies insisted that the color set could be included in the program

## Usage
### Before model training it is necessary to train word embeddings:
```
$ python train_word_embeddings.py
```

This script train word embeddings defined in [Mikolov et al. Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546)

#### Parameters:
`--use-cuda`

`--num-iterations`

`--batch-size`

`--num-sample` –– number of sampled from noise tokens


### To train model use:
```
$ python train.py
```

#### Parameters:
`--use-cuda`

`--num-iterations`

`--batch-size`

`--learning-rate`
 
`--dropout` –– probability of units to be zeroed in decoder input

`--use-trained` –– use trained before model

### To sample data after training use:
```
$ python sample.py
```
#### Parameters:
`--use-cuda`

`--num-sample`

