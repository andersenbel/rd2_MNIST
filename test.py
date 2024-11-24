import argparse
import os  # Для роботи з файловою системою
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import ModelA, ModelB, ModelC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Перевірка наявності GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використовується пристрій: {device}", flush=True)

# Парсинг аргументів командного рядка
parser = argparse.ArgumentParser(description="Тестування моделі")
parser.add_argument(
    "--model", type=str, choices=["A", "B", "C"], default="B",
    help="Виберіть модель: A, B або C (за замовчуванням B)"
)
parser.add_argument(
    "--model-path", type=str, default="mnist_model.pth",
    help="Шлях до файлу збереженої моделі (за замовчуванням mnist_model.pth)"
)
args = parser.parse_args()

# Вибір моделі
if args.model == "A":
    model = ModelA()
    hidden_layers = 1
elif args.model == "B":
    model = ModelB()
    hidden_layers = 2
elif args.model == "C":
    model = ModelC()
    hidden_layers = 3

model = model.to(device)

# Завантаження ваг моделі
model.load_state_dict(torch.load(
    args.model_path, map_location=device, weights_only=True))
model.eval()
print(f"Модель {args.model} ({
      hidden_layers} прихованих шарів) завантажена для тестування.", flush=True)

# Завантаження даних
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(
    root='./datasets', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Тестування
correct = 0
total = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Розрахунок точності
accuracy = 100 * correct / total
print(f"Модель {args.model} ({hidden_layers} прихованих шарів), Точність: {
      accuracy:.2f}%", flush=True)

# Матриця невідповідностей
conf_matrix = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=list(range(10)))
disp.plot(cmap='Blues')

# Створення папки для результатів
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)

# Збереження матриці невідповідностей у файл
visualization_path = os.path.join(
    results_folder, f"confusion_matrix_{args.model}.png")
plt.title(f"Матриця невідповідностей для моделі {
          args.model}\nТочність: {accuracy:.2f}%")
plt.savefig(visualization_path)
print(f"Матриця невідповідностей збережена у файл: {
      visualization_path}", flush=True)

# Збереження результатів у текстовий файл
results_path = os.path.join(results_folder, "results.txt")
with open(results_path, "a") as f:
    f.write(f"Модель {args.model} ({hidden_layers} прихованих шарів): Точність: {
            accuracy:.2f}%\n")
print(f"Результати збережено у файл: {results_path}", flush=True)