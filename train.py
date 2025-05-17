import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

from torchvision.models import VGG16_Weights, ResNet18_Weights


def get_data_loaders(data_dir):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    valid_data = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=valid_transforms)

    return (train_data,
            torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True),
            torch.utils.data.DataLoader(valid_data, batch_size=32),
            train_data.class_to_idx)


def build_model(arch='vgg16', hidden_units=512):
    if arch == 'vgg16':
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        input_units = model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    elif arch == 'resnet18':
        model = models.resnet18(ResNet18_Weights.DEFAULT)
        input_units = model.fc.in_features
        classifier = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.fc = classifier
    else:
        raise ValueError("Supported architectures: vgg16, resnet18")

    return model

def train_model(model, train_loader, valid_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    params = model.classifier.parameters() if hasattr(model, 'classifier') else model.fc.parameters()
    optimizer = optim.Adam(params, lr=lr)

    model.to(device)

    for e in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {e + 1}/{epochs}.. "
              f"Train loss: {running_loss / len(train_loader):.3f}.. "
              f"Valid loss: {valid_loss / len(valid_loader):.3f}.. "
              f"Valid accuracy: {accuracy / len(valid_loader):.3f}")
        model.train()


def save_checkpoint(model, path, class_to_idx, arch='vgg16', hidden_units=4096):
    if hasattr(model, 'classifier'):
        classifier = model.classifier
    else:
        classifier = model.fc

    checkpoint = {
        'architecture': arch,
        'class_to_idx': class_to_idx,
        'classifier': classifier,
        'state_dict': model.state_dict(),
        'hidden_units': hidden_units,
    }
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', default='checkpoint.pth')
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    device = torch.device("mps" if args.gpu and torch.backends.mps.is_available() else "cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    train_data, train_loader, valid_loader, class_to_idx = get_data_loaders(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    model.class_to_idx = class_to_idx

    train_model(model, train_loader, valid_loader, device, args.epochs, args.learning_rate)
    save_checkpoint(model, args.save_dir, class_to_idx, args.arch, args.hidden_units)


if __name__ == '__main__':
    main()