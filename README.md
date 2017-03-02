# Pytorch Recurrent Variational Autoencoder 

## Model:
This is the implementation of Samuel Bowman's [Generating Sentences from a Continuous Space] (https://arxiv.org/abs/1511.06349#)
with Kim's [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615) embedding for tokens

## Sampling examples:
- > the new machine could be used to increase the number of ventures block in the company 's \<unk> shopping system to finance diversified organizations

- > u.s. government officials also said they would be willing to consider whether the proposal could be used as urging and programs

- > men believe they had to go on the \<unk> because their \<unk> were \<unk> expensive important

- > the companies insisted that the color set could be included in the program

## Usage
### Before model training it is neccesary to train word embeddings:
```
$ cd model/utils
$ python train_word_embeddings.py
```

### To train model use:
```
$ cd model
$ python train.py
```

### To sample data after training use:
```
$ cd model
$ python sample.py
```

