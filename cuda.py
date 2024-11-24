import torch

# Перевіряємо наявність GPU
if torch.cuda.is_available():
    print("GPU доступний!")
    device = torch.device("cuda")  # Використовуємо GPU
else:
    print("GPU недоступний, використовується CPU.")
    device = torch.device("cpu")  # Використовуємо CPU
