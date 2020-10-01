# %%
import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from trains import Task

# %%
task = Task.init(project_name='Trains Test Examples', task_name='Simple Net')

# %%
training_data_path = './data/cats-vs-dogs-small/train'
validation_data_path = './data/cats-vs-dogs-small/validation'
test_data_path = './data/cats-vs-dogs-small/validation'

# %%
transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# %%
training_data = torchvision.datasets.ImageFolder(
    root=training_data_path,
    transform=transforms
)

# %%
validation_data = torchvision.datasets.ImageFolder(
    root=validation_data_path,
    transform=transforms
)

# %%
test_data = torchvision.datasets.ImageFolder(
    root=test_data_path,
    transform=transforms
)

# %%
batch_size = 64
training_data_loader = data.DataLoader(training_data, batch_size=batch_size)
validation_data_loader = data.DataLoader(validation_data, batch_size=batch_size)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)


# %%
# noinspection PyAbstractClass
class SimpleNet(nn.Module):
    image_size = 3 * 64 * 64

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(SimpleNet.image_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, SimpleNet.image_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


# %%
simplenet = SimpleNet()

# %%
optimizer = optim.Adam(simplenet.parameters(), lr=0.0005)


# %%
def run_training_loop(model=simplenet, loss_fn=nn.CrossEntropyLoss(), epochs=30, device='cpu'):
    for epoch in range(epochs):
        training_loss = 0.0
        validation_loss = 0.0
        num_items = 0
        model.train()
        for batch in training_data_loader:
            optimizer.zero_grad()
            inputs, target = batch
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
            num_items += len(target)
        training_loss /= num_items

        model.eval()
        num_correct = 0
        num_examples = 0
        num_items = 0
        for batch in validation_data_loader:
            inputs, target = batch
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = loss_fn(output, target)
            validation_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output, dim=0), dim=1)[1], target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
            num_items += len(target)
        validation_loss /= num_items

        print(f'Epoch: {epoch}, '
              f'tr-loss: {training_loss:.2f}, val-loss: {validation_loss:.2f}, '
              f'accuracy: {num_correct / num_examples}')


# %%
run_training_loop(simplenet)

