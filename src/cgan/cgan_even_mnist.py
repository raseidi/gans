import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
plt.style.use('ggplot')

torch.manual_seed(111)
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

n_classes = 2
class CDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(n_classes, 10)
        
        self.model = nn.Sequential(
            nn.Linear(784+10, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out #.squeeze()

class CGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(n_classes, 10)
        
        self.model = nn.Sequential(
            nn.Linear(100+10, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 1, 28, 28)

''' READING DATA '''
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])
train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)

''' LOADING DATA LOADER AND MODELS '''
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

discriminator = CDiscriminator().to(device=device)
generator = CGenerator().to(device=device)
# x = torch.randn((batch_size, 100))
# y = torch.randint(0, 2, (batch_size, 1)).reshape(-1)
# generator(x, y)
# for real_samples, mnist_labels in train_loader:
#     break
# mnist_labels.apply_(lambda x: 0 if x % 2 != 0 else 1)
# discriminator(real_samples, mnist_labels)


lr = 1e-4
num_epochs = 100
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

''' TRAINING LOOP '''
import numpy as np
g_epoch_loss = []
d_epoch_loss = []

for epoch in range(num_epochs):
    D_loss_list, G_loss_list = 0, 0
    for n, (mnist_images, mnist_labels) in enumerate(train_loader):
        batch_size = mnist_images.shape[0]
        mnist_images = mnist_images.to(device=device)
        mnist_labels = mnist_labels.apply_(lambda x: 0 if x % 2 != 0 else 1).to(device=device)
        real_labels = torch.ones((batch_size,1)).to(device=device)
        fake_labels = torch.zeros((batch_size, 1)).to(device=device)
        real_fake_labels = torch.cat((real_labels, fake_labels)).to(device=device)

        # training discriminator
        optimizer_discriminator.zero_grad()
        z = torch.randn((batch_size, 100)).to(device=device)
        gen_images = generator(z.detach(), mnist_labels) 

        all_samples = torch.cat((mnist_images, gen_images))
        all_labels = torch.cat((mnist_labels, mnist_labels))

        predicted_labels = discriminator(all_samples, all_labels) # classifying true/fake
        d_loss = loss_function(predicted_labels, real_fake_labels)
        d_loss.backward()
        optimizer_discriminator.step()
        D_loss_list += d_loss.item()

        # training generator
        optimizer_generator.zero_grad()
        gen_images = generator(z.detach(), mnist_labels)

        predicted_labels = discriminator(gen_images, mnist_labels) 
        g_loss = loss_function(predicted_labels, real_labels) # trying to fool the discriminator
        g_loss.backward()
        optimizer_generator.step()
        G_loss_list += g_loss.item()

    print(f"Epoch: {epoch}\t Loss D.: {D_loss_list / n} Loss G.: {G_loss_list / n}")
    g_epoch_loss.append(G_loss_list/n)
    d_epoch_loss.append(D_loss_list/n)
    if np.mean(G_loss_list) < 0.1:
        break

z = torch.randn(16, 100).cuda()
# labels = torch.arange(16).apply_(lambda x: 0 if x % 2 != 0 else 1).cuda()
labels = torch.tensor([*torch.zeros(8).int(), *torch.ones(8).int()]).cuda()
images = generator(z, labels).cpu().detach()
torch.save(generator, 'models/cgan_v1')
# model = torch.load('models/cgan_v1')
# model.eval()
# images = model(z, labels).cpu().detach()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
plt.show()

plt.figure(dpi=100)
plt.style.use('ggplot')
plt.plot(np.arange(len(g_epoch_loss)), g_epoch_loss, '-')
plt.plot(np.arange(len(d_epoch_loss)), d_epoch_loss, '-')
plt.legend(['Generator loss', 'Discriminator loss'])
# plt.yscale('log')
plt.savefig('models/cgan_v1_loss.pdf', dpi=100)
plt.show()