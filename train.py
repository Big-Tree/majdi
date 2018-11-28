import time
import torch
import copy
import matplotlib.pyplot as plt
import usefulFunctions as uf

def train_model(model, criterion, optimizer, num_epochs, device, datasets,
                dataloader, save_dir, run_num, early_stopping=99999,
                show_plots=True, save_plots=True, scheduler=None):
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
                  'train_loss':69,
                  'train_acc':0,
                  'val_acc':0,
                  'val_loss':69,
                  'test_acc':0,
                  'test_loss':69}
    if save_plots or show_plots:
        if show_plots:
            plt.ion()
        f0 = plt.figure()
        f1 = plt.figure()
    epoch = 0
    while (epoch - best_model['epoch'] < early_stopping and
            epoch < num_epochs):
        print('({})Epoch {}/{}'.format(run_num, epoch, num_epochs - 1))
        print('-' * 10)
        start_epoch_time = time.time()

        # Each epoch has a training and validation phase
        last_epoch = {'acc':0, 'loss':0}
        last_last_epoch = dict(last_epoch)
        for phase in ['train', 'test', 'val']:
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
            print('    {} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # keep best model
            if phase == 'val' and epoch_acc > best_model['val_acc']:
                best_model['val_acc'] = epoch_acc.item()
                best_model['val_loss'] = epoch_loss
                best_model['test_acc'] = last_epoch['acc']
                best_model['test_loss'] = last_epoch['loss']
                best_model['train_acc'] = last_last_epoch['acc']
                best_model['train_loss'] = last_last_epoch['loss']
                best_model['model'] = copy.deepcopy(model.state_dict())
                best_model['epoch'] = epoch
            if save_plots or show_plots:
                # Plot loss
                plt.figure(f0.number)
                plt.cla() #Clear axis
                plt.title('Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.axvline(x=best_model['epoch'],  linestyle='dashed',
                            color='k')
                plt.plot(stats['losses']['train'],
                    label='train ({:.2f})'.format(
                        best_model['train_loss']))
                plt.plot(stats['losses']['test'],
                    label='test ({:.2f})'.format(
                        best_model['test_loss']))
                plt.plot(stats['losses']['val'],
                    label='val ({:.2f})'.format(
                        best_model['val_loss']))
                plt.legend()
                if show_plots:
                    plt.pause(0.001)
                # Plot accuracy
                plt.figure(f1.number)
                plt.cla() #Clear axis
                plt.title('Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.grid(True)
                plt.axvline(x=best_model['epoch'],  linestyle='dashed',
                            color='k')
                plt.plot(stats['acc']['train'], label='train ({:.2f})'.format(
                    best_model['train_acc']))
                plt.plot(stats['acc']['test'], label='test ({:.2f})'.format(
                    best_model['test_acc']))
                plt.plot(stats['acc']['val'], label='val ('+'{:.2f})'.format(
                        best_model['val_acc']))
                plt.legend()
                if show_plots:
                    plt.pause(0.001)

                last_last_epoch = dict(last_epoch)
                last_epoch['acc'] = epoch_acc
                last_epoch['loss'] = epoch_loss

        epoch += 1
        print('  Epoch time: {:.2f}s'.format(time.time()-start_epoch_time))
        print()
    if save_dir != None and save_plots:
        # Save figures
        uf.save_matplotlib_figure(
            save_dir, f0, 'svg', '(' + str(run_num) + ')loss')
        uf.save_matplotlib_figure(
            save_dir, f1, 'svg', '(' + str(run_num) + ')accuracy')

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_model['val_acc']))
    model.load_state_dict(best_model['model'])
    plt.close('all')
    return model
