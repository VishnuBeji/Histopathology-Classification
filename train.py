import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

def train_cnn(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train_predictions = 0
        total_train_predictions = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train_predictions += labels.size(0)
            correct_train_predictions += (predicted == labels).sum().item()

        train_accuracy = correct_train_predictions / total_train_predictions

        model.eval()
        correct_val_predictions = 0
        total_val_predictions = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val_predictions += labels.size(0)
                correct_val_predictions += (predicted == labels).sum().item()

        val_accuracy = correct_val_predictions / total_val_predictions

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: {running_loss / len(train_loader)}, "
              f"Train Accuracy: {train_accuracy}, "
              f"Validation Accuracy: {val_accuracy}")

def train_vt(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and compute accuracy
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Validation loop
        model.eval()
        total_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.logits, 1)
                total_val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

        # Calculate and print epoch statistics
        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples
        val_accuracy = total_val_correct / total_val_samples

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

