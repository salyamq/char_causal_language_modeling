from datasets import load_dataset

dataset = load_dataset("tiny_shakespeare")


def get_data(text: str) -> str:
    """получаем датасет: трейн, для валидации и тестовый"""

    if text == 'train':
        return dataset['train'][0]['text']
    elif text == 'validation':
        return dataset['validation'][0]['text']
    elif text == 'test':
        return dataset['test'][0]['text']
    else:
        print('Напиши верный тип!')

train_dataset = get_data("train")

# сортируем лист
chars = sorted(list(set(train_dataset)))
vocab_size = len(chars) # размерность словаря: 64

stoi = {ch: i for i, ch in enumerate(chars)} # стринг в айди
itos = {i: ch for i, ch in enumerate(chars)} # айди в стринг

