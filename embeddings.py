import torch
import torch.nn as nn
import tokenizer


stoi = tokenizer.stoi # словарь

train_dataset = tokenizer.get_data('train') # трейн датасет
validation_dataset = tokenizer.get_data('validation')  # валидейшн датасет
test_dataset = tokenizer.get_data('test') # тест датасет

vocab_size = tokenizer.vocab_size # размерность словаря
d_model = 128 # размерность эмбеддинга



# переводим текст в токены, т.е F -> 16 и так далее, возвращаем тензор
train_ids = torch.tensor([stoi[ch] for ch in train_dataset]) # ndim = 1
val_ids = torch.tensor([stoi[ch] for ch in validation_dataset]) # ndim = 1
test_ids = torch.tensor([stoi[ch] for ch in test_dataset]) # ndim = 1


