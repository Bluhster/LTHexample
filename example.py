import torch
import torch.optim
import torch.utils.data
import torch.nn as nn

from model import densenet121
import helper

epochs = 5
lr = 0.1
momentum = 0.9
weight_decay = 1e-4
batch_size = 64


model_unpruned = densenet121()

initial_model_dict = model_unpruned.state_dict()

model_gmp = densenet121()
model_gmp_it = densenet121()
model_gmp_it.load_state_dict(model_unpruned.state_dict())

def compute(model, num_epochs = 5, sparsity = 0.8, iterative_pruning = False):
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    
    training_loss = []
    training_acc =[]

    test_loss = []
    test_acc = []

    train_data = helper.CIFAR10Dataset(root = './data', train = True, download = True)
    test_data = helper.CIFAR10Dataset(root = './data', train = False, download = False)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size = batch_size, 
                                               shuffle = True,
                                                )
    
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size = batch_size, 
                                               shuffle = False,
                                                )
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr,
                                momentum = momentum,
                                weight_decay = weight_decay)
    
    to_prune, all_layers = helper.get_layers(model)
    
    if iterative_pruning:
                                                # epochs -1 because after the last epoch pruning won't be applied
        sparsity = 1 - (1 - sparsity) ** (1 / (num_epochs-1))
    
    for epoch in range(num_epochs):
        print("Starting epoch ", epoch+1)
        loss, acc = train(train_loader, model, criterion, optimizer)

        training_loss.append(loss.item())
        training_acc.append(acc)

        t_loss, t_acc = test(test_loader, model, criterion)

        test_loss.append(t_loss.item())
        test_acc.append(t_acc)
    
    helper.prune_net(sparsity, to_prune)
    return training_loss, training_acc, test_loss, test_acc


def train(train_loader, model, criterion, optimizer):

    model.train()
    num_correct_pred = 0

    for i, (input, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        else:
            input = input.cpu()
            target = target.cpu()

        output = model(input)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = 0
    return loss, acc

def test(test_loader, model, criterion):

    model.eval()
    num_correct_pred = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            else:
                input = input.cpu()
                target = target.cpu()

            output = model(input)

            loss = criterion(output, target)

    acc = 0
    return loss, acc

training_loss, training_acc, test_loss, test_acc = compute(model_gmp, num_epochs=30, sparsity=0.8, iterative_pruning=False)

print('training losses/accs: ', training_loss, training_acc)
print('test losses/accs: ', test_loss, test_acc)