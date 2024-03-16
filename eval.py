import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#TO evaluate Vision Transformer
def evaluate_vt_model(model, test_loader, criterion, device):
    model.eval()
    test_labels = []
    test_predictions = []
    test_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            probabilities = torch.softmax(outputs.logits, dim=1)
            _, predicted = torch.max(outputs.logits, 1)
            
            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(predicted.cpu().numpy())
            test_probabilities.extend(probabilities[:, 1].cpu().numpy())

    return test_labels, test_probabilities, test_predictions

#To evaluate CNN models
def evaluate_cnn_model(model, test_loader, criterion, device):
    model.eval()
    test_labels = []
    test_predictions = []
    test_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(predicted.cpu().numpy())
            test_probabilities.extend(probabilities[:, 1].cpu().numpy())

    return test_labels, test_probabilities, test_predictions


#To print accuracy, F1 score, AUC etc and the confusion matrix
def calculate_metrics(test_labels, test_probabilities, test_predictions):
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions)
    recall = recall_score(test_labels, test_predictions)
    auc = roc_auc_score(test_labels, test_probabilities)
    conf_matrix = confusion_matrix(test_labels, test_predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"AUC: {auc}")

    # Plot confusion matrix
    plt.figure(figsize=(3, 2))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['SSA', 'HP'], yticklabels=['SSA', 'HP'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    return auc

def visualize_cam(model, test_images, test_labels, num_samples):
    # Specify the target layers for Grad-CAM
    target_layers = [model.layer4[-1]] 

    cam = GradCAM(model=model, target_layers=target_layers)

    random_indices = random.sample(range(len(test_images)), num_samples)
    num_images = len(random_indices)

    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3.5*num_images))

    model.eval()
    for i, image_index in enumerate(random_indices):
        input_tensor = test_images[image_index].unsqueeze(0) 

        # Forward pass the preprocessed image through the model
        output = model(input_tensor)
        _, predicted_label = torch.max(output, 1)

        # Specify the target for generating the CAM
        #targets = [ClassifierOutputTarget(test_labels[image_index])]  
        targets = [ClassifierOutputTarget(predicted_label)]  

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        grayscale_cam = grayscale_cam[0, :]

        normalized_img = (test_images[image_index].detach().cpu().numpy() - test_images[image_index].detach().cpu().numpy().min()) / (test_images[image_index].detach().cpu().numpy().max() - test_images[image_index].detach().cpu().numpy().min())
        normalized_img = normalized_img.transpose(1, 2, 0)

        visualization = show_cam_on_image(normalized_img, grayscale_cam, use_rgb=True)

        # Display the original image
        axes[i, 0].imshow(np.transpose(test_images[image_index].cpu().numpy(), (1, 2, 0)))
        axes[i, 0].set_title(f"Original Image\nTrue Label: {test_labels[image_index]}")

        # Display the CAM overlay
        axes[i, 1].imshow(visualization)
        axes[i, 1].set_title(f"Class Activation Map\n Predicted Label: {predicted_label[0]}")

    plt.tight_layout()
    plt.show()

def visualize_cam_vt(vit_model, test_images, test_labels, num_samples):
    # Specify the target layers for Grad-CAM
    target_layers = [vit_model.vit.encoder.layer[-1].output]

    cam = GradCAM(model=vit_model, target_layers=target_layers)

    random_indices = random.sample(range(len(test_images)), num_samples)
    num_images = len(random_indices)

    # Create subplots for each image
    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3.5*num_images))

    vit_model.eval()
    for i, image_index in enumerate(random_indices):
        input_tensor = test_images[image_index].unsqueeze(0)  

        output = vit_model(input_tensor)
        _, predicted_label = torch.max(output.logits, 1)

        # Specify the target for generating the CAM
        targets = [ClassifierOutputTarget(test_labels[image_index])]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        grayscale_cam = grayscale_cam[0].squeeze().detach().cpu().numpy()

        normalized_img = (test_images[image_index].cpu().numpy() - test_images[image_index].cpu().numpy().min()) / (test_images[image_index].cpu().numpy().max() - test_images[image_index].cpu().numpy().min())
        visualization = show_cam_on_image(normalized_img, grayscale_cam, use_rgb=True)

        # Display the original image
        axes[i, 0].imshow(np.transpose(test_images[image_index].cpu().numpy(), (1, 2, 0)))
        axes[i, 0].set_title(f"Original Image\nTrue Label: {test_labels[image_index]}")

        # Display the CAM overlay
        axes[i, 1].imshow(visualization)
        axes[i, 1].set_title(f"Class Activation Map\n Predicted Label: {predicted_label.item()}")

    plt.tight_layout()
    plt.show()

