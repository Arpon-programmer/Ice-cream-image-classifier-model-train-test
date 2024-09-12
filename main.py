import torch
import torchvision.transforms as T
from PIL import Image

trans = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

classes = ['bj', 'breyers', 'hd', 'talenti']

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
model = torch.jit.load('Ice_cream_image_model_100%.pt', map_location= 'mps')
model.eval()

def prediction(img_path, model):
    img = Image.open(img_path).convert('RGB')
    trans_img = trans(img)
    with torch.no_grad():
        out = model(trans_img.unsqueeze(0).to(device))
        _, pred = torch.max(out, dim=1)
        return classes[pred.item()]

def operation():
    image_path = input("Give Image path : ")
    result = prediction(image_path, model)
    print(f"Predicted Ice Cream type: {result}")

while True:
    try:
        operation()
    except FileNotFoundError:
        print("File not found. Please provide valid image path.")
        operation()