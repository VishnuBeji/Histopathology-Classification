# eda.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import seaborn as sns
import cv2

def visualize_class_distribution(class_counts):
    plt.figure(figsize=(7, 6))

    plt.subplot(2, 2, 1)
    plt.bar(class_counts.index, class_counts.values)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Train Class Distribution (Bar Plot)')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Train Class Distribution (Pie Chart)')
    plt.axis('equal')  
    plt.tight_layout()
    plt.show()

def upsample_minority_class(train_images, train_labels_tensor, train_labels, minority_class_label, device):
    minority_class_indices = np.where(train_labels == minority_class_label)[0]
    minority_class_images = [train_images[i] for i in minority_class_indices]
    minority_class_labels = train_labels[minority_class_indices]

    # Calculate the number of samples needed to match the majority class
    # This will be total - 2* minority class
    majority_class_count = len(train_labels) - 2 * len(minority_class_labels)

    # Randomly select minority class samples and apply data augmentation to match the majority class count
    # We augment by applying random rotation or mirror transformation
    augmented_minority_class_images = []
    augmented_minority_class_labels = []

    while len(augmented_minority_class_images) < majority_class_count:
        # Randomly select a minority class sample
        idx = np.random.randint(len(minority_class_images))
        image = minority_class_images[idx]

        transformation = np.random.choice(['rotate', 'mirror'])
        if transformation == 'rotate':
            angle = np.random.randint(-15, 15)  # Random rotation angle between -15 and 15 degrees
            image = transforms.functional.rotate(image, angle)
        elif transformation == 'mirror':
            image = transforms.functional.hflip(image)

        augmented_minority_class_images.append(image)
        augmented_minority_class_labels.append(minority_class_labels[idx])

    augmented_minority_class_images_tensor = [image.to(device) for image in augmented_minority_class_images]
    augmented_minority_class_labels_tensor = torch.tensor(augmented_minority_class_labels, dtype=torch.long).to(device)

    upsampled_train_images = train_images + augmented_minority_class_images_tensor
    upsampled_train_labels = torch.cat((train_labels_tensor, augmented_minority_class_labels_tensor))

    return upsampled_train_images, upsampled_train_labels


def plot_vote_distribution(train_df, test_df):
    palette = {'SSA': '#ff0000', 'HP': '#0069b1'}

    fig, axes = plt.subplots(ncols=2, figsize=(8, 3))

    sns.histplot(data=train_df, x='Number of Annotators who Selected SSA (Out of 7)', hue='Majority Vote Label',
                 palette=palette, multiple='stack', binwidth=1, binrange=(-0.5, 7.5), ax=axes[0])
    axes[0].set_title('Train Data')
    axes[0].set_xlabel('Number of Votes for SSA (Out of 7)')
    axes[0].set_ylabel('Count')

    sns.histplot(data=test_df, x='Number of Annotators who Selected SSA (Out of 7)', hue='Majority Vote Label',
                 palette=palette, multiple='stack', binwidth=1, binrange=(-0.5, 7.5), ax=axes[1])
    axes[1].set_title('Test Data')
    axes[1].set_xlabel('Number of Votes for SSA (Out of 7)')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

# Function to perform CLAHE
def clahe(image_tensor):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    image_np_uint8 = (image_np * 255).astype(np.uint8)

    lab_image = cv2.cvtColor(image_np_uint8, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl_l_channel = clahe.apply(l_channel)

    # Merge CLAHE-enhanced L channel with original A and B channels
    clahe_lab_image = cv2.merge((cl_l_channel, a_channel, b_channel))

    # Convert back to RGB color space
    clahe_rgb_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2RGB)

    # Convert NumPy array back to tensor
    clahe_image_tensor = torch.tensor(clahe_rgb_image, dtype=torch.float32).permute(2, 0, 1) / 255.0

    return clahe_image_tensor