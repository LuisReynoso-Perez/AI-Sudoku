#Import required packages
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

# Set the seed for Python's random module
random_seed = 42
random.seed(random_seed)

# Set the seed for PyTorch
torch_seed = 42
torch.manual_seed(torch_seed)

#Load Data. Split into Training Data and Test data and create dataloaders for each.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = len(train_data)
test_size = len(test_data)

train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data,batch_size=1, shuffle=False)

for images, labels in train_loader:
    # Visualize the first image in the batch
    plt.imshow(images[0].squeeze(), cmap='gray')
    plt.title(f"Label: {labels[0]}")
    plt.show()
    break  # Break after visualizing the first image in the batch

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Input channels (1 for grayscale), output channels (6 filters), kernel size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Adjust based on input size after conv2. 
        #Implicit that the pooling layer is also applied after the 2nd Conv Layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output: 10 class probabilities (0-9)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # Flatten for FC layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)  # Log Softmax for better numerical stability
        return x

model = MNISTClassifier()  # Create an instance of the defined model class
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 10

for epoch in range(num_epochs):
    # Train loop
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # Forward pass
        #Start your code
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #End your Code
        
        # Backpropagation
        
        #Start your code
        optimizer.zero_grad()
        loss.backward()
        #End your Code

        # Update weights in accordance with optimizer
        #Start your code
        optimizer.step()
        #End your code

        # Print statistics
        running_loss += loss.item()
        
        
        # if (i==937):
        #    print('[%d] loss: %.3f' %
        #          (epoch + 1, running_loss/938))
        #    running_loss = 0.0
            
running_loss = 0.0
concatenated_outputs = []


for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    with torch.no_grad():  
        #begin your code, calculate loss in "loss"
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        concatenated_outputs.append(outputs)
        #End your code
        # Print statistics
        running_loss += loss.item()
        #if i % 1000 == 999:  # Print every 1000 samples
           
final_output = torch.cat(concatenated_outputs, dim=0)

random_seed = 5 #Don't change the random seed
random.seed(random_seed)
# Randomly choose three images
random_indices = random.sample(range(len(test_loader)), 3)

for i, data in enumerate(test_loader, 0):
    inputs, labels = data 
    for j in random_indices:
        if (i==j):
            image_data = inputs
            label = labels
            #Start your code, first obtain class probabilities using torch.nn.functional.softmax from logits stored in final_output in the previous cell. Then use torch.argmax to get the predicted class.             
            class_probabilities = torch.nn.functional.softmax(final_output[i], dim=0)
            prediction = torch.argmax(class_probabilities).item()
            #End your code
            print(f"Image at index {i} - Predicted Label: {prediction}")
            plt.imshow(image_data.squeeze(), cmap='gray')
            plt.title(f"Label: {label}")
            plt.show()