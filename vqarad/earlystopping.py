
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms



# Model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=786, out_features=130),
            nn.ReLU(),
            nn.Linear(in_features=130, out_features=66),
            nn.ReLU(),
            nn.Linear(in_features=66, out_features=12),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)


# Train
def traindata(device, model, epochs, optimizer, loss_function, train_loader, valid_loader):
    # Early stopping
    last_loss = 100
    patience = 2
    triggertimes = 0

    for epoch in range(1, epochs+1):
        model.train()

        for times, data in enumerate(train_loader, 1):
            input = data[0].to(device)
            label = data[1].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward and backward propagation
            output = model(input.view(input.shape[0], -1))
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            # Show progress
            if times % 100 == 0 or times == len(train_loader):
                print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, epochs, times, len(train_loader), loss.item()))

        # Early stopping
        current_loss = validation(model, device, valid_loader, loss_function)
        print('The Current Loss:', current_loss)

        if current_loss > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return model

        else:
            print('trigger times: 0')
            trigger_times = 0

        last_loss = current_loss

    return model


def validation(model, device, valid_loader, loss_function):

    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        for data in valid_loader:
            input = data[0].to(device)
            label = data[1].to(device)

            output = model(input.view(input.shape[0], -1))
            loss = loss_function(output, label)
            loss_total += loss.item()

    return loss_total / len(valid_loader)


def test(device, model, test_loader):

    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            input = data[0].to(device)
            label = data[1].to(device)

            output = model(input.view(input.shape[0], -1))
            _, predicted = torch.max(output.data, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()

    print('Accuracy:', correct / total)


def main():
    # GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device state:', device)

    epochs = 100
    batch_size = 66
    lr = 0.004
    loss_function = nn.NLLLoss()
    model= Net()  ### you can not use the same name before and after assignment 
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    # Data
    trainset = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
    testset = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
   
    trainset_size = int(len(trainset) * 0.8)
    validset_size = len(trainset) - trainset_size
    trainset, validset = data.random_split(trainset, [trainset_size, validset_size])

    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    validloader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)

    # Train
    model = traindata(device, model, epochs, optimizer, loss_function, trainloader, validloader)

    # Test
    test(device, model, testloader)


if __name__ == '__main__':
    main()
    print("heelo")