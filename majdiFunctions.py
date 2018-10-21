import math
import numpy as np

# This function will return the average accuracy for the entire data that it is
# sent.  Can calculate accuracy for entire epoch.  Set batch size so that it
# can all fit in the GPU
def get_stats_epoch(net, criterion,  data_all, labels_all, batch_size):
    print('    length data: ', len(data_all))
    iterations = math.ceil(len(data_all)/batch_size)
    all_accuracies = []
    all_losses = []
    size_of_final_accuracy = 0
    for i in range(iterations):
        output = net(data_all[i*batch_size : (i+1)*batch_size])
        labels = labels_all[i*batch_size : (i+1)*batch_size]
        maxOutput = np.asarray(
            [np.argmax(_) for _ in output.data.cpu().numpy()])
        pred = np.zeros((len(output), 2))
        pred[range(len(output)), maxOutput] = 1
        accuracy = sum(pred == labels.cpu().numpy())/len(labels)
        loss = criterion(output, labels)
        all_losses.append(loss.item())
        accuracy = accuracy[0]
        all_accuracies.append(accuracy)
        size_of_final_accuracy = len(output)

    out_accuracy = (sum(all_accuracies[0:-1])/len(all_accuracies)
        + all_accuracies[-1] / len(all_accuracies)
            * size_of_final_accuracy/batch_size)
    out_loss = (sum(all_losses[0:-1])/len(all_losses)
        + all_losses[-1] / len(all_losses)
            * size_of_final_accuracy/batch_size)
    print('    out_loss: ', out_loss)
    #print('    final loss: ', all_losses[-1])
    print(all_losses, '\n')
    return out_accuracy, out_loss
