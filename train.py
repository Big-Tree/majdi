import time
import torch
import copy
import matplotlib.pyplot as plt

def train_model(model, criterion, optimizer, num_epochs, device, datasets,
                dataloader, scheduler=None):
    start_time = time.time()
    best_acc = 0.0
    # Figures
    f0 = plt.figure()
    f1 = plt.figure()
    plt.figure(f0.number) # Change selected figure
    #plt.axvline(x=model_best['step'],  linestyle='dashed', color='k')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.figure(f1.number)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        start_epoch_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for dataloader_dict in dataloader[phase]:
                inputs = dataloader_dict['image']
                labels = dataloader_dict['label']
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                optimizer.zero_grad()

                # We only need to track the history in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, 1)
                    loss = criterion(outputs, labels)
                    tmp = (preds == torch.argmax(labels.data, 1))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds ==
                                              torch.argmax(labels.data, 1))

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = (running_corrects.double() /
                         len(datasets[phase]))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # keep best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        

        print()
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
