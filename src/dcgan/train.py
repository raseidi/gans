from pendulum import time
import time
import torch
from torch import nn

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms


# from src.dcgan.dcgan import DCDiscriminator, DCGenerator, initialize_weights
from dcgan import DCDiscriminator, DCGenerator, initialize_weights
plt.style.use('ggplot')

torch.manual_seed(111)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
LR = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1 # for mnist
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],
        [0.5 for _ in range(CHANNELS_IMG)]
    )
])

dataset = torchvision.datasets.MNIST(root='.', train=True, transform=transforms, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = DCGenerator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
disc = DCDiscriminator(CHANNELS_IMG, FEATURES_DISC).to(DEVICE)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
# writter_real = SummaryWriter('logs/real')
# ...

gen.train()
disc.train()

criterion = nn.BCELoss()

G_loss_list, D_loss_list = [], []

start = time.time()
for epoch in range(NUM_EPOCHS):
    G_loss, D_loss = 0, 0
    for n, (real, _) in enumerate(loader):
        real = real.to(DEVICE)
        z = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(DEVICE)
        fake = gen(z)

        # train disc: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True) # reuse z for generator
        opt_disc.step()

        # train gen: min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        D_loss += loss_disc.detach()
        G_loss += loss_gen.detach()

    print(f'Epoch {epoch}: D_loss={D_loss/n:.4f} \t G_loss={G_loss/n:.4f}')
    G_loss_list.append(G_loss/n)
    D_loss_list.append(D_loss/n)

end = time.time() - start
print(f'Elapsed time: {end/60:.2f} min')

z = torch.randn((16, Z_DIM, 1, 1)).to(DEVICE)
images = gen(z).cpu().detach()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].reshape(64, 64), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
plt.show()