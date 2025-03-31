import torch
from sklearn.model_selection import train_test_split

def split_dataset(pth_path, train_ratio=0.9, output_dir="."):
    # Load the dataset
    dataset = torch.load(pth_path)
    data = dataset["data"]  # Tensor of image data
    labels = dataset["labels"]  # Tensor of corresponding labels

    # Ensure the data and labels have the same length
    assert len(data) == len(labels), "Data and labels must have the same length."

    # Generate train-test split indices
    train_indices, test_indices = train_test_split(
        range(len(labels)), train_size=train_ratio, random_state=42, shuffle=True
    )

    # Split the data and labels using the indices
    train_data = data[train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_data = data[test_indices]
    test_labels = [labels[i] for i in test_indices]

    # Save the splits as separate .pth files
    train_dataset = {"data": train_data, "labels": train_labels}
    test_dataset = {"data": test_data, "labels": test_labels}

    train_pth_path = f"{output_dir}/train_data.pth"
    test_pth_path = f"{output_dir}/test_data.pth"

    torch.save(train_dataset, train_pth_path)
    torch.save(test_dataset, test_pth_path)

    print(f"Training dataset saved to: {train_pth_path}")
    print(f"Testing dataset saved to: {test_pth_path}")


# Usage Example
# Path to the original dataset
pth_dir = "../data.pth"

# Directory where the splits will be saved
output_dir = "../"

# Call the function to split the dataset
split_dataset(pth_dir, train_ratio=0.9, output_dir=output_dir)

raw_data = torch.load("../test_data.pth")
print(type(raw_data["labels"]))
print(type(raw_data["data"]))