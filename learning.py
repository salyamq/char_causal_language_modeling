import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from model import TransformerCLM, model, get_device
from data_preparation import train_loader, val_loader, test_loader
from embeddings import vocab_size, d_model


# гиперпараметры
lr = 3e-4
epochs = 10
device = get_device()

model = model.to(device) # переводим в cuda/mps если есть
optimizer = optim.AdamW(params = model.parameters(), # параметры модели
                        lr = lr, # шаг
                        weight_decay = 0.01, # аля l2 регуляризация
                        betas=(0.9, 0.95)) # моментумы: скорость и адаптивность

loss_func = nn.CrossEntropyLoss() # функция ошибки




best_val = float("inf")  # сохраняем лучшее значение val loss

for epoch in range(epochs):
    model.train() # переводим в обучение
    total_loss = 0 # общая ошибка

    loop = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{epochs}") # для отслеживания

    for x, y in loop:
        x = x.to(device) # переводим в cuda/mps если есть
        y = y.to(device) # переводим в cuda/mps если есть

        optimizer.zero_grad() # обнуляем градиенты
        predicted = model(x) # предсказываем следущий токен (bath_size, context, vocab_size)

        # flatten для CrossEntropyLoss
        loss = loss_func(
            predicted.view(-1, predicted.size(-1)),
            y.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # чтобы не взорвались веса
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}: train loss = {total_loss / len(train_loader):.4f}")

    # ВАЛИДАЦИЯ
    model.eval() # переводим в eval режим
    with torch.inference_mode():
        val_loss = 0

        loop_val = tqdm(val_loader, desc="Validation")

        for i, (x_val, y_val) in enumerate(loop_val):
            x_val = x_val.to(device) # переводим в cuda/mps если есть
            y_val = y_val.to(device) # переводим в cuda/mps если есть

            predicted_val = model(x_val) # предсказываем следующий токен

            loss = loss_func(
                predicted_val.view(-1, predicted_val.size(-1)),
                y_val.view(-1)
            )

            val_loss += loss.item()

            loop_val.set_postfix(loss=loss.item()) # лосс на каждой итерации

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: val loss = {avg_val_loss:.4f}")

    # СОХРАНЯЕМ ЛУЧШУЮ МОДЕЛЬ
    if avg_val_loss < best_val:
        best_val = avg_val_loss
        torch.save(model.state_dict(), "best_model.pt")

    model.train() # возвращаем обратно в train режим

# сохраняем модель (финальная версия)
torch.save(model.state_dict(), "last_model.pt")