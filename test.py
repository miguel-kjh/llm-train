import torch
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
