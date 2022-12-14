import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os


def generate_constants(steps, initial=1e-4, final=2e-2):
    """
    Generates a linear schedule from [initial, final].
    Returns (alpha, beta), where beta is the linear schedule and alpha is 1 - beta
    """

    beta = torch.linspace(initial, final, steps)
    alpha = 1 - beta

    sqrt_alpha = torch.sqrt(alpha)

    # See Eq. 4 (DDPM) for formulation of alpha bar
    alpha_bar = torch.cumprod(alpha, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    # See Eq. 6 (DDPM) for formulation of beta tilde
    alpha_bar_prev = torch.cat((torch.tensor([1]), alpha_bar[:-1]))
    beta_tilde = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)

    return (
        alpha,
        beta,
        sqrt_alpha,
        alpha_bar,
        sqrt_alpha_bar,
        sqrt_one_minus_alpha_bar,
        beta_tilde,
    )


def get_index_from_list(vals, t, x_shape):
    # Helper function taken from DeepFindr's video (see README)
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward(x_0, t):
    """
    Generates a noised image :math:`x_t` given an :math:`x_0`.
    .. math::
        x_t = \\sqrt{\\bar{\\alpha}_t}\\ x_0 + \sqrt{1 - \\bar{\\alpha}_t}\\ \\varepsilon
    where :math:`\\varepsilon \\sim \\mathcal{N}(0, I)`.
    :param x_0: torch.Tensor: A 4D batch tensor (B, C, W, H)
    :param t: torch.Tensor: A 1D tensor of time steps to generate
    :return: (xt, E), A tuple where xt is the result of the forward process after t, and E is the noise applied onto the image.
    :rtype: (torch.Tensor, torch.Tensor)
    """

    A_t = get_index_from_list(sqrt_alpha_bar, t, x_0.shape).to(device)
    stdev = get_index_from_list(sqrt_one_minus_alpha_bar, t, x_0.shape).to(device)

    E = torch.randn_like(x_0).to(device)
    x_t = (x_0.to(device) * A_t) + (stdev * E)

    return (x_t, E)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        inputs = 6 + 3 * len(kernels)

        self.stack = nn.Sequential(
            nn.Linear(inputs, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.stack(x)
        return x


def create_dataloader(image_size=(256, 256), batch_size=5):
    transform = transforms.Compose(
        # Resize images in dataset to image_size and normalise color values to [-1, 1]
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    traindata = datasets.CelebA(
        "./data", split="train", transform=transform, download=True
    )
    testdata = datasets.CelebA(
        "./data", split="test", transform=transform, download=True
    )
    validdata = datasets.CelebA(
        "./data", split="valid", transform=transform, download=True
    )

    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=batch_size, shuffle=True, drop_last=True
    )
    validloader = torch.utils.data.DataLoader(
        validdata, batch_size=batch_size, shuffle=True, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return (trainloader, validloader, testloader)


def sampler(T, I):
    # Assumption: T is [B, C, H, W], and I = image_samples * (x, y) = (2, image_samples)
    sampled_T = T[..., I[0], I[1]]
    sampled_T = sampled_T.transpose(1, -1)
    nshape = sampled_T.shape
    sampled_T = sampled_T.reshape((nshape[0] * nshape[1], nshape[2]))

    return sampled_T


def Loss(model, x_0, t, I):
    x_t, E = forward(x_0, t)

    # Gather values of x_t and E at indices I
    sampled_x_t = sampler(x_t, I)
    sampled_E = sampler(E, I)

    # Apply Gaussian blur onto x_0 and sample
    noise = [transforms.GaussianBlur(*kernel)(x_0) for kernel in kernels]
    sampled_noise = [sampler(N, I).to(device) for N in noise]

    # Normalise I to [-1, 1]
    I = (2.0 * I - image_size) / image_size

    # Normalise t to [-1, 1]
    t = (2.0 * t - steps) / steps

    I = I.permute((1, 0)).repeat(batch_size, 1)
    t = t[:, None].repeat(image_samples, 1)
    ivec = torch.cat((t, I, sampled_x_t, *sampled_noise), dim=1)

    # Predict E given ivec
    predicted_E = model(ivec)

    return F.mse_loss(predicted_E, sampled_E)


def save_model():
    torch.save(model.state_dict(), "model.pth")


def train(train, validate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("-" * 32)
        print(f"Epoch {epoch + 1}")
        print("-" * 32)

        for step, (x_0, _) in enumerate(train):
            optimizer.zero_grad()

            x_0 = x_0.float()
            I = torch.randint(0, image_size, (2, image_samples), device=device)
            t = torch.randint(0, steps, (batch_size,), device=device).long()

            loss = Loss(model, x_0, t, I)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"\tstep {step}, loss {loss:>.8f}")

                if step % 250 == 0:
                    validation_loss = validate_model(validate)
                    print(f"validation {epoch+1}/{step}: {validation_loss:>.8f}")

                    if step > 0 and step % 500 == 0:
                        save_model()


@torch.no_grad()
def validate_model(loader):
    total_loss = 0

    for (x_0, _) in loader:
        I = torch.randint(0, image_size, (2, image_samples), device=device)
        t = torch.randint(0, steps, (batch_size,), device=device).long()
        x_0 = x_0.float().to(device)

        total_loss += Loss(model, x_0, t, I).item()

    loss = total_loss / len(loader)

    return loss


@torch.no_grad()
def denoise(x, t):
    """
    Given an image :math:`x_t`, denoise the image using the model to find :math:`x_{t - 1}`.
    From Eq. 10 in DDPM:
    .. math::
        \\mu_\\theta = \\frac{1}{\\sqrt{\\bar{\\alpha_t}}}\\left(x_t - \\frac{\\beta_t}{\\sqrt{1 - \\bar{\\alpha}_t}}\\varepsilon\\right)
    Using :math:`\\mu_\\theta`, we can then reconstruct :math:`x_{t - 1}`.
    """
    beta_t = get_index_from_list(beta, t, x.shape)
    sqrt_one_minus_alpha_bar_t = get_index_from_list(
        sqrt_one_minus_alpha_bar, t, x.shape
    )
    alpha_sqrt_t = get_index_from_list(sqrt_alpha, t, x.shape)

    # Sample X at every point
    I = torch.cartesian_prod(*(torch.arange(image_size),) * 2).permute((1, 0))
    sampled_x_t = sampler(x, I)

    # Apply Gaussian blur onto x_0 and sample
    noise = [transforms.GaussianBlur(*kernel)(x) for kernel in kernels]
    sampled_noise = [sampler(N, I) for N in noise]

    # Normalise I to [-1, 1]
    I_hat = (2.0 * I - image_size) / image_size

    # Normalise t to [-1, 1]
    t_hat = (2.0 * t - steps) / steps

    I_hat = I_hat.permute((1, 0))
    t_hat = t_hat[:, None].repeat(image_size * image_size, 1)

    ivec = torch.cat((t_hat, I_hat, sampled_x_t, *sampled_noise), dim=1).to(device)

    E_theta = model(ivec)
    E_theta = E_theta.detach().cpu()
    E_theta = E_theta.reshape((x.shape[0], image_size * image_size, x.shape[1]))
    E_theta = E_theta.transpose(1, -1)
    E_theta = E_theta.reshape((x.shape[0], x.shape[1], image_size, image_size))

    mu_theta = (x - beta_t * E_theta / sqrt_one_minus_alpha_bar_t) / alpha_sqrt_t

    if t == 0:
        return mu_theta

    noise = torch.randn_like(x)
    beta_tilde_t = get_index_from_list(beta_tilde, t, x.shape)
    predicted = mu_theta + torch.sqrt(beta_tilde_t) * noise

    return predicted


if __name__ == "__main__":
    batch_size = 64
    image_size = 128
    image_samples = 8192
    steps = 1000
    epochs = 100
    learning_rate = 1e-2
    kernels = ((3, 0.5), (9, 1.5), (27, 4.5), (81, 13.5))

    assert image_samples <= image_size * image_size

    (
        alpha,
        beta,
        sqrt_alpha,
        alpha_bar,
        sqrt_alpha_bar,
        sqrt_one_minus_alpha_bar,
        beta_tilde,
    ) = generate_constants(steps=steps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP().to(device)

    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))

    if True:
        (trainloader, validloader, testloader) = create_dataloader(
            (image_size, image_size), batch_size
        )
        train(trainloader, validloader)
    else:
        with torch.no_grad():
            image = torch.randn((3, image_size, image_size))
            # image = next(iter(trainloader))[0][0]
            image = torch.squeeze(image)

            step = steps // 10
            x = image[None, ...]
            images = []
            # images = [torch.squeeze(x.cpu())]

            # for idx in range(steps, step):
            # t = torch.tensor([idx]).long().cpu()
            # x, E = forward(x, t)

            # image = torch.squeeze(x.detach().cpu())
            # images.append(image)

            for idx in reversed(range(steps)):
                t = torch.tensor([idx]).long().cpu()
                x = denoise(x.cpu(), t)

                if idx % step == 0:
                    image = x.detach().cpu()
                    image = torch.squeeze(image)
                    images.append(image)

            grid = make_grid(images, nrow=10, normalize=True, value_range=(-1, 1))
            grid = grid.permute(1, 2, 0).numpy()

            plt.imshow(grid)
        plt.show()
