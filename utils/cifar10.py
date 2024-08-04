import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


def load_cifar10_dataset(batch_size, thresholds):
    trans = [transforms.ToTensor()]
    if thresholds == 3:
        trans.append(transforms.Lambda(lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0)))
    elif thresholds == 31:
        trans.append(transforms.Lambda(lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0)))
    transform = transforms.Compose(trans)
    train_set = CIFAR10('./data-cifar10', train=True, download=True, transform=transform)
    test_set = CIFAR10('./data-cifar10', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                                              drop_last=True)
    input_dim_dataset = 3 * 32 * 32 * thresholds
    num_classes = 10
    return train_loader, test_loader, input_dim_dataset, num_classes
