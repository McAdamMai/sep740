import os
import tarfile
from io import BytesIO
import requests
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

# Define the path
output_dir = "../tools_image"
pth_dir = "../data.pth"
test_dir = "../test_data.pth"
train_dir = "../train_data.pth"


# Define the classes
desired_classes1 = {
    "hammer": "n03481172"
}
desired_classes = {
    "hammer": "n03481172",
    "screwdriver": "n04133789",
    "wrench": "n04517823",
    "drill": "n03208938",
    "chainsaw": "n03000684"
}
#https://image-net.org/data/winter21_whole/n03000684.tar
# Define a preprocessing data pipeline
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Download and process images for each class
def download_tar(desired_classes, output_dir):
    for class_name, synset_id in desired_classes.items():
        # Create a folder for this class
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        tar_file_path = os.path.join(class_dir, "images.tar")
        print(f"Downloading images for class '{class_name}'...")
        if not os.path.exists(tar_file_path):
        # ImageNet URL for the tar file
            url = f"https://image-net.org/data/winter21_whole/{synset_id}.tar"
        # Send a request to fetch the tar file
            try:
                response = requests.get(url, timeout=10, stream=True)
                if response.status_code == 200:
                    print("Downloading .tar file...")
                    with open(tar_file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Downloaded .tar file to {tar_file_path}")
                else:
                    print(f"Failed to download .tar file. Status code: {response.status_code}")
            except Exception as e:
                print(f"Failed to fetch image URLs for class '{class_name}': {e}")

        else:
            print("Tar file existed")
        extracted_dir = os.path.join(class_dir, "extracted_images")
        os.makedirs(extracted_dir, exist_ok=True)
        if not os.listdir(extracted_dir): # if folder is empty, begin extraction
            print(f"extracting {class_name} file to {extracted_dir}...")
            with tarfile.open(tar_file_path, "r") as tar:
                tar.extractall(path=extracted_dir)
            print(f"{class_name} file extracted")
        else:
            print("images had been loaded")

def convert2set(desired_classes, output_dir):
    processed_data = []
    labels = []
    paths = [os.path.join(output_dir, file, "extracted_images")
             for file in os.listdir(output_dir)
             if os.path.isdir(os.path.join(output_dir, file))
             ]
    print("Processing image")
    for image_dir in paths:
        for root, _, files in os.walk(image_dir):
            for file in tqdm(files, desc="resizing images"):
                try:
                    image_path = os.path.join(root,file)
                    image = Image.open(image_path).convert("RGB")
                    processed_image = transform(image)
                    processed_data.append(processed_image)
                    labels.append(int(desired_classes[os.path.basename(os.path.dirname(root))][-3:]))
                except Exception as e:
                    print(f"failed to process the imageL: {file} {e} ")
    return processed_data, labels

def set2torch(images, labels, pth_dir):
    dataset_tensor = torch.stack(images)
    #labels_tensor = torch.tensor(labels, dtype=torch.int)
    dataset = {
        "data": dataset_tensor,
        "labels": labels
    }
    torch.save(dataset,pth_dir)
    print("Data tensor shape:", dataset_tensor.shape)
    print("Labels tensor:", len(labels))

image_dir = os.path.join(os.path.join(output_dir, "hammer"),"extracted_images")
images, labels = convert2set(desired_classes=desired_classes, output_dir=output_dir)
set2torch(images=images, labels=labels, pth_dir=pth_dir)

