import time
import torch
import copy
import matplotlib.pyplot as plt

def train_model(model, criterion, optimizer, num_epochs, device, datasets,
                dataloader, scheduler=None):
    start_time = time.time()
    # Figures
    stats = {'losses':{
                 'train':[],
                 'val':[],
                 'test':[]},
             'acc':{
                 'train':[],
                 'val':[],
                 'test':[]}}
    best_model = {'model':None,
                  'epoch':0,
                  'acc':0,
                  'loss':69}
    f0 = plt.figure()
    f1 = plt.figure()
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
            stats['losses'][phase].append(epoch_loss)
            stats['acc'][phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # keep best model
            if phase == 'val' and epoch_acc > best_model['acc']:
                best_model['acc'] = epoch_acc.item()
                best_model['loss'] = epoch_loss
                best_model['model'] = copy.deepcopy(model.state_dict())
                best_model['epoch'] = epoch
            # Plot loss
            plt.figure(f0.number)
            plt.cla() #Clear axis
            plt.title('Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.axvline(x=best_model['epoch'],  linestyle='dashed', color='k')
            plt.plot(stats['losses']['train'], label='train')
            plt.plot(stats['losses']['val'], label='val ('+'{:.2f}'.format(
                best_model['loss'])+')')
            plt.legend()
            plt.pause(0.001)
            # Plot accuracy
            plt.figure(f1.number)
            plt.cla() #Clear axis
            plt.title('TrainingAccuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.axvline(x=best_model['epoch'],  linestyle='dashed', color='k')
            plt.plot(stats['acc']['train'], label='train')
            plt.plot(stats['acc']['val'], label='val ('+'{:.2f}'.format(
                    best_model['acc']) + ')')
            plt.legend()
            plt.pause(0.001)
        

        print()
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_model['acc']))
