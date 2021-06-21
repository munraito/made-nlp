from torchtext.legacy.data import Field, TabularDataset
from nltk.tokenize import WordPunctTokenizer
from transformers import AutoTokenizer

import random

tokenizer_W = WordPunctTokenizer()
tokenizer_bert = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")


def tokenize(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())


def tokenize_reverse(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())[:-1]


def tokenize_for_bert(x, tokenizer=tokenizer_bert, **kwargs):
    return tokenizer.encode(x, truncation=True, max_length=512, add_special_tokens=False, **kwargs)


def process_dataset(path_to_data, reverse_trg=False, bf=False):
    SRC = Field(tokenize=tokenize, 
                init_token = '<sos>', eos_token = '<eos>', lower = True, batch_first=bf) # , batch_first = True
    
    if reverse_trg:
        TRG = Field(tokenize=tokenize_reverse,
                init_token = '<sos>',  eos_token = '<eos>', lower = True, batch_first=bf)
    else:
        TRG = Field(tokenize=tokenize,
                init_token = '<sos>',  eos_token = '<eos>', lower = True, batch_first=bf)
    
    # fixing seed for reproducibility
    random.seed(420)
    dataset = TabularDataset(
        path=path_to_data, format='tsv', fields=[('trg', TRG), ('src', SRC)]
    )
    
    train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    
    SRC.build_vocab(train_data, min_freq = 3)
    TRG.build_vocab(train_data, min_freq = 3)
    print(f"Unique tokens in source (ru) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
    
    return train_data, valid_data, test_data, SRC, TRG


def process_dataset_bert(path_to_data):
    pad_index = tokenizer_bert.convert_tokens_to_ids(tokenizer_bert.pad_token)
    
    SRC = Field(
        use_vocab=False, tokenize=tokenize_for_bert, pad_token=pad_index
    )
    TRG = Field(
        tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True
    )
    
    random.seed(420)
    dataset = TabularDataset(
        path=path_to_data, format='tsv', fields=[('trg', TRG), ('src', SRC)]
    )
    
    train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    
    SRC.build_vocab(train_data, min_freq = 3)
    TRG.build_vocab(train_data, min_freq = 3)
    print(f"Unique tokens in source (ru) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
    
    return train_data, valid_data, test_data, SRC, TRG
