import torch
import math
def seccion(message):
    l=len(message)*2
    s=len(message)//2
    c = " "*s
    print("="*l)
    print("{}{}".format(c, message))
    print("="*l)
     
#CREAR TENSORES
seccion("Tensor Vacio")
x = torch.empty(3, 4)#(FILAS,COLUMNAS)
print(type(x))
print(x)

seccion("Tensor with zeros")
zeros = torch.zeros(2, 3)#(FILAS,COLUMNAS)
print(zeros)

seccion("Tensor with ones")
ones = torch.ones(2, 3)#(FILAS,COLUMNAS)
print(ones)

seccion("Tensor with Random Seed")
torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)

#TENSORES ALEATORIOS Y SEMILLAS
seccion("TENSOR RANDOM AND SEEDS")
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1730)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)

#FORMAS DE LOS TENSORES
seccion("FORMAS DE LOS TENSORES")

x= torch.empty(2, 2, 3)
print(x.shape)
print(x)
empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)
seccion(" ")

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)
seccion(" ")

ones_like_x =torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)
seccion(" ")

random_like_x = torch.rand_like(x)
print(random_like_x.shape)
print(random_like_x)

#OTRA FORMA DE CREAR UN TENSOR DEFINIENDO LOS VALORES
seccion("TENSOR DEFINIENDO LOS VALORES")
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)

#TIPOS DE DATOS DE TENSORES
seccion("TIPOS DE DATOS DE TENSORES")

a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c =b.to(torch.int32)
print(c)

#Matematicas y logica con Tensores PyTorch
seccion("Matematicas y logica con Tensores PyTorch")

ones =torch.zeros(2, 2) + 1
print(ones)

twos = torch.ones(2, 2) * 2
print(twos)

threes = (torch.ones(2 ,2) * 7 -1)/2
print(threes)

fours = twos ** 2
print(fours)

sqrt2s = twos ** 0.5
print(sqrt2s)

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = fours + ones
print(fives)

dozens = threes * fours
print(dozens)

    #Error intencional al operar con tensores de distintas formas
"""
a = torch.rand(2, 3)
b = torch.rand(3, 2)
print(a * b)
"""

#BROADCASTING EN TENSORES
seccion("BROADCASTING EN TENSORES")

rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)

a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 3ra dimension & 2da identicas a las de a, dim 1 ausente
print(b)

c = a * torch.rand(   3, 1) # 3ra dim = 1, 2da dim identica a la de a
print(c)

d = a * torch.rand(   1, 2) # 3ra dim identica a la de a, 2da dim = 1
print(d)

#OPERACIONES CON TENSORES 
seccion("OPERACIONES CON TENSORES ")
seccion("common functions")

# common functions
a = torch.rand(2, 4) * 2 - 1
print(a)
print(torch.abs(a))
print(torch.ceil(a))# devuelve el entero más pequeño que es mayor o igual 
print(torch.floor(a))# devuelve el entero más grande que es menor o igual
print(torch.clamp(a, -0.5, 0.5))# limita los valores a un rango

seccion("trigonometric ")
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print(angles)

sines = torch.sin(angles)
print (sines)

inverses = torch.asin(sines)
print(inverses)

# bitwise operations
seccion("bitwise operations")
print("Bitwise XOR:")

b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# comparisons:
seccion("comparisons")
print('Broadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # many comparison ops support broadcasting!
print(torch.eq(d, e)) # returns a tensor of type bool

seccion("reductions")
# reductions:
print(torch.max(d))        # returns a single-element tensor
print(torch.max(d).item()) # extracts the value from the returned tensor
print(torch.mean(d))       # average
print(torch.std(d))        # standard deviation
print(torch.prod(d))       # product of all numbers
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # filter unique elements

seccion("vector and linear algebra operations")
# vector and linear algebra operations
v1 = torch.tensor([1., 0., 0.])         # x unit vector
v2 = torch.tensor([0., 1., 0.])         # y unit vector
m1 = torch.rand(2, 2)                   # random matrix
m2 = torch.tensor([[3., 0.], [0., 3.]]) # three times identity matrix

seccion("Vectors & Matrices")
print(torch.linalg.cross(v2, v1)) # negative of z unit vector (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)
print(m3)                  # 3 times m1
print("""\nU: Matriz ortogonal de vectores singulares izquierdos. 
S: Vector con los valores singulares ordenados de mayor a menor.
V: Matriz ortogonal de vectores singulares derechos.""")

print(torch.svd(m3))       # singular value decomposition

#ALTERACION DE TENSORES EN SU LUGAR
seccion("ALTERACION DE TENSORES EN SU LUGAR")
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # esta operación crea un nuevo tensor en la memoria
print(a)              # a no ha cambiado

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # note el guión bajo
print(b)              # b ha cambiado

a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('\nBefore:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)
print("---")
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)                # el contenido de c ha cambiado

assert c is d           # se fija si c & d son el mismo objeto, no que solo contienen los mismos valore
assert id(c), old_id    # se asegura que el nuevo c sea el mismo que el viejo

torch.rand(2, 2, out=c) # funciona también para constructores
print(c)                # c ha cambiado nuevamente
assert id(c), old_id    # todavía es el mismo objeto

#COPIAR TENSORES
seccion("COPIAR TENSORES")

a = torch.ones(2, 2)
b = a

a[0][1] = 561
print(b)

a = torch.ones(2, 2)
b = a.clone()

assert b is not a 
print(torch.eq(a, b))

a[0][1] = 561
print(b)

#MANIPULACION DE LA FORMA DEL TENSOR
seccion("MANIPULACION DE LA FORMA DEL TENSOR")

a = torch.rand(3, 266, 266)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)

c = torch.rand(1, 1, 1, 1, 1)
print(c)

a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)

#recordando el siguiente ejemplo de broadcasting
a = torch.ones(4, 3, 2)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)

a = torch.ones(4, 3, 2)
b = torch.rand(   3)     # intentar multiplicar a * b dará un error de tiempo de ejecución
c = b.unsqueeze(1)       # cambiar a un tensor bidimensional, agregando un nuevo dim al final
print(c.shape)
print(a * c)             # ¡El broadcast funciona de nuevo!

batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)

output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)

#PUENTE CON NUMPY
seccion("PUENTE CON NUMPY")

import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)
###
pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)
###
numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)

#DATASETS & DATALOADERS
seccion("DATASETS & DATALOADERS")

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
"""
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
#plt.show()
"""
#AutoGrad
seccion("AUTOGRAD")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
print(a)

b = torch.sin(a)
#plt.plot(a.detach(), b.detach())
#plt.show()
print(b)

c = 2 * b
print(c)

d = c + 1
print(d)

out = d.sum()
print(out)

print('d:')
print(d.grad_fn)
print(d.grad_fn.next_functions)
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
print('\nc:')
print(c.grad_fn)
print('\nb:')
print(b.grad_fn)
print('\na:')
print(a.grad_fn)


out.backward()
print(a.grad)
plt.plot(a.detach(), a.grad.detach())
plt.show()