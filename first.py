import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
import numpy as np

# Генерация данных
def generate_data(n_samples=1000, x_min=0, x_max=50, limit=100, step_min=1, step_max=10):
    data = []
    for _ in range(n_samples):
        x0 = random.uniform(x_min, x_max)
        step = random.uniform(step_min, step_max)
        steps_needed = int((limit - x0) / step)
        data.append([x0, step, steps_needed])
    return data, limit

# Данные
data, limit = generate_data(2000)
data = np.array(data)
X = data[:, :2]  # x0, step
y = data[:, 2]   # steps_needed

# Нормализация
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X = (X - X_mean) / X_std
y = y.reshape(-1, 1)

# Трен/тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Модель
class StepPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.model(x)

model = StepPredictor()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Обучение
for epoch in range(200):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Пример пользовательского значения
example_x0 = 25.0
example_step = 4.2

# Нормализуем для модели
example_input = torch.tensor([[
    (example_x0 - X_mean[0]) / X_std[0],
    (example_step - X_mean[1]) / X_std[1]
]], dtype=torch.float32)

# Предсказание
model.eval()
with torch.no_grad():
    predicted_steps = model(example_input).item()

true_steps = int((limit - example_x0) / example_step)

print(f"\n📊 Реальные шаги до предела: {true_steps}")
print(f"🤖 Предсказанные моделью шаги: {predicted_steps:.1f}")

# --------- АНИМАЦИЯ ---------
x_values = [example_x0]
while x_values[-1] < limit:
    x_values.append(x_values[-1] + example_step)
time_steps = list(range(len(x_values)))

fig, ax = plt.subplots()
line, = ax.plot([], [], 'bo-', label='Значение параметра')
limit_line = ax.axhline(y=limit, color='r', linestyle='--', label='Предел')

ax.set_xlim(0, len(x_values))
ax.set_ylim(example_x0 - 5, limit + 10)
ax.set_title("Анимация приближения к предельному значению")
ax.set_xlabel("Шаг")
ax.set_ylabel("Значение")
ax.legend()

def update(frame):
    line.set_data(time_steps[:frame+1], x_values[:frame+1])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(x_values), interval=400, repeat=False)

plt.show()