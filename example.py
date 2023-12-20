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
        loss, acc = train(train_loader, model, criterion, optimizer)

        training_loss.append(loss)
        training_acc.append(acc)

        t_loss, t_acc = test(test_loader, model, criterion)

        test_loss.append(t_loss)
        test_acc.append(t_acc)
    
    helper.prune_net(sparsity, to_prune)
    return training_loss, training_acc, test_loss, test_acc


def train(train_loader, model, criterion, optimizer):

    model.train()
    num_correct_pred = []

    for i, (input, target) in enumerate(train_loader):

        output = model(input)

        loss = criterion(output, target)
        
        # print(output)
        # #print(target[i])
        # if output.idxmax() == target[i]-1:
        #     num_correct_pred.append(1)
        # else:
        #     num_correct_pred.append(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()
    
    #acc = (len(train_loader)/sum(num_correct_pred)).float()
    acc = 0

    return loss, acc

def test(test_loader, model, criterion):

    model.eval()
    num_correct_pred = []

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):

            output = model(input)

            loss = criterion(output, target)

            # if output == target[i]:
            #     num_correct_pred.append(1)
            # else:
            #     num_correct_pred.append(0)

    #acc = (len(test_loader)/sum(num_correct_pred)).float()
    acc = 0

    return loss, acc

training_loss, training_acc, test_loss, test_acc = compute(model_gmp, 10, 0.8)

print('right hereee: ', training_loss)
print('asdfkjasödkfjqpöwoierhjtgiuhghsdg: ', test_loss)