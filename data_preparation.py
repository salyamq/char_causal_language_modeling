from torch.utils.data import Dataset, DataLoader
from embeddings import train_ids, val_ids, test_ids

class ShakespeareDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data # данные
        self.block_size = block_size # длина одной последовательности (окна)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size] # cрез от idx до idx+self.block_size
        y = self.data[idx+1:idx+self.block_size+1] # тот же срез, но сдвиг на +1
        return x, y

block_size = 256 # 256 символов контекста
batch_size = 32

train_loader = DataLoader(ShakespeareDataset(train_ids, block_size),
                          batch_size=batch_size,
                          shuffle=True) # (32, 256)
val_loader = DataLoader(ShakespeareDataset(val_ids, block_size),
                        batch_size=batch_size,
                        shuffle=False) # (32, 256)
test_loader = DataLoader(ShakespeareDataset(test_ids, block_size),
                         batch_size=batch_size,
                         shuffle=False) # (32, 256)