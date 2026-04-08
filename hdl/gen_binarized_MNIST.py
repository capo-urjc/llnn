import torch
from torchvision import datasets, transforms

# Transforms: resize to 20x20 and convert to tensor
transform = transforms.Compose([
    transforms.Resize((20,20)),
    transforms.ToTensor()
])

# Load dataset (test set)
dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

with open("images.txt", "w") as f_img, open("labels.txt", "w") as f_lbl:

    for image, label in loader:

        # image: [1,1,20,20]
        image = image * 255

        # binarization
        image_bin = (image >= 127).int()

        # flatten to 400
        img_flat = image_bin.view(-1).tolist()

        # convert to string
        img_str = "".join(str(p) for p in img_flat)

        f_img.write(img_str + "\n")
        f_lbl.write(str(label.item()) + "\n")