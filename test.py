"""import torch
import torch.nn as nn

# Creamos un tensor con las dimensiones de tu entrada original: [10, 20, 30, 100]
x = torch.randn(10, 20, 30, 100)

# Para aplicar convolución 2D, necesitamos reorganizar las dimensiones a [batch_size, channels, height, width]
# Podemos interpretar el 20 como canales, y 30x100 como las dimensiones espaciales
# Reorganizamos la entrada a: [batch_size=10, channels=20, height=30, width=100]
# x = x.permute(0, 1, 2, 3)  # Asegurarnos de que está en el orden adecuado
print(x.shape)  # Verifica que es [10, 20, 30, 100]

# Definimos una convolución 2D que reducirá las 10 instancias a una única salida
# Usamos una convolución con in_channels=20 (tu segunda dimensión)
# Si queremos mantener el tamaño espacial, usamos kernel_size=1 y stride=1
conv2d = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1)

# Aplicamos la convolución
output = conv2d(x)

# Si quieres quitar la primera dimensión, puedes usar un promedio o max pooling sobre la dimensión del batch.
output = torch.mean(output, dim=0)  # Promedia sobre el batch size (10)

# La salida será de tamaño [20, 30, 100] como esperas
print(output.shape)  # Verifica que es [20, 30, 100]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# Transformaciones para los datos de CIFAR-10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Descargar y preparar el conjunto de datos CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Definir la capa condicional con distancia angular
class ConditionalLayer(nn.Module):
    def __init__(self, input_size, threshold=0.99, max_iter=5):
        super(ConditionalLayer, self).__init__()
        self.fc = nn.Linear(input_size, input_size)
        self.threshold = threshold  # Umbral de distancia angular
        self.max_iter = max_iter    # Máximo número de iteraciones para reprocesar la capa

    def angular_distance(self, x, x_prime):
        dot_product = torch.sum(x * x_prime, dim=-1)
        norm_x = torch.norm(x, dim=-1)
        norm_x_prime = torch.norm(x_prime, dim=-1)
        cos_theta = dot_product / (norm_x * norm_x_prime)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        return cos_theta

    def forward(self, x):
        iteration = 0
        while iteration < self.max_iter:
            x_prime = self.fc(x)
            cos_theta = self.angular_distance(x, x_prime)
            
            if torch.mean(cos_theta) >= self.threshold:
                break
            else:
                x = x + x_prime
            
            iteration += 1
        
        return x_prime

# Definir una MLP que use la capa condicional
class ConditionalMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, threshold=0.5, max_iter=5):
        super(ConditionalMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Capa condicional que decide si pasar a la siguiente capa
        self.cond_fc = nn.Linear(hidden_size, 1)  # Salida escalar para la condición
        self.threshold = threshold
        self.max_iter = max_iter

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        
        iteration = 0
        while iteration < self.max_iter:
            x_prime = self.fc2(x)  # Procesar con fc2
            x_prime = torch.relu(x_prime)

            # Evaluar la condición con una capa lineal adicional
            condition = torch.sigmoid(self.cond_fc(x_prime))  # Obtener un valor entre 0 y 1
            
            # Si el valor es mayor que el umbral, continuar al siguiente paso
            if torch.mean(condition) >= self.threshold:
                break
            else:
                # Si no se cumple la condición, retroalimentar x para volver a procesar
                x = x + x_prime  # Sumar la salida al input original
            
            iteration += 1

        # Finalmente, aplicar la capa final para la clasificación
        x = self.fc3(x_prime)
        return x
    
    
class ConfidenceBasedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, entropy_threshold=0.5, max_iter=5):
        super(ConfidenceBasedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Modificar para no reducir a 10 aquí
        self.fc3 = nn.Linear(hidden_size, output_size)  # Capa final para la clasificación
        self.softmax = nn.Softmax(dim=1)
        self.entropy_threshold = entropy_threshold
        self.max_iter = max_iter

    def calculate_entropy(self, probs):
        """Calcular la entropía de la distribución de probabilidad"""
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # Evitar log(0)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)

        iteration = 0
        while iteration < self.max_iter:
            x_prime = self.fc2(x)  # Procesar con fc2 pero manteniendo la dimensión 256
            x_prime = torch.relu(x_prime)

            # Ahora aplicar softmax y calcular la entropía para medir la incertidumbre
            logits = self.fc3(x_prime)  # Pasar x_prime por la capa final de clasificación
            probs = self.softmax(logits)  # Obtener la distribución de probabilidad

            # Calcular la entropía
            entropy = self.calculate_entropy(probs)

            # Si la entropía es menor que el umbral, pasar a la siguiente capa
            if torch.mean(entropy) < self.entropy_threshold:
                break
            else:
                # Si la entropía es alta, retroalimentar la salida de x_prime para volver a pasar por fc2
                x = x + x_prime  # Sumar x_prime a x para refinar la entrada

            iteration += 1
        
        return logits  # Devuelve los logits finales después del bucle

class RefinementMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_iter=3):
        super(RefinementMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.refiner = nn.Linear(hidden_size, hidden_size)  # Capa de refinamiento
        self.fc3 = nn.Linear(hidden_size, output_size)  # Capa final para la clasificación
        self.max_iter = max_iter

    def forward(self, x):
        # Procesamiento inicial con fc1
        x = self.fc1(x)
        x = torch.relu(x)

        # Iteraciones de refinamiento en fc2
        x_prime = self.fc2(x)
        x_prime = torch.relu(x_prime)
        
        # Bucles de refinamiento: Refinamos la salida de x_prime
        for _ in range(self.max_iter):
            # Refinar x_prime a través de la capa refiner
            refined = torch.relu(self.refiner(x_prime))
            # Sumar la salida refinada a la salida actual para mejorar la representación
            x_prime = x_prime + refined

        # Capa final de clasificación
        logits = self.fc3(x_prime)
        return logits

# Modelo condicional para CIFAR-10
class CIFAR10ConditionalMLP(nn.Module):
    def __init__(self, threshold=0.99):
        super(CIFAR10ConditionalMLP, self).__init__()
        self.flatten = nn.Flatten()
        # baseline 52.08
        #self.mlp = ConditionalMLP(32*32*3, 256, 10, threshold) # 50
        # self.mlp = ConfidenceBasedMLP(32*32*3, 256, 10) # 51,3
        # self.mlp = RefinementMLP(32*32*3, 256, 10, max_iter=10) # 51,3


    def forward(self, x):
        x = self.flatten(x)
        x = self.mlp(x)
        return x

# Configuración del modelo, la pérdida y el optimizador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CIFAR10ConditionalMLP(threshold=0.99).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
def train(model, trainloader, criterion, optimizer, device):
    model.train()
    for epoch in tqdm(range(5), desc="Trainig"):  # Número de épocas
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward y optimización
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

# Evaluación del modelo
def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')

# Entrenar y evaluar
train(model, trainloader, criterion, optimizer, device)
test(model, testloader, device)
