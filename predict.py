import argparse
import torch
from PIL import Image
import json
from torchvision import transforms
from train import build_model  # re-use architecture helper

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    model = build_model(arch = checkpoint['architecture'], hidden_units=checkpoint['hidden_units'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dim

def predict(image_path, model, device, topk=5):
    model.eval()
    model.to(device)

    img = process_image(image_path)
    img = img.to(device)

    with torch.no_grad():
        output = model(img)

    probs, indices = torch.exp(output).topk(topk, dim=1)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in indices]

    return probs, classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "mps" if args.gpu and torch.mps.is_available() else "cpu")

    model = load_checkpoint(args.checkpoint)
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(args.image_path, model, device, args.top_k)
    names = [cat_to_name.get(str(c), c) for c in classes]

    for i in range(len(names)):
        print(f"{names[i]}: {probs[i]*100:.2f}%")

if __name__ == '__main__':
    main()
