import math
import numpy as np
from sklearn import metrics

# This function will return the average accuracy for the entire data that it is
# sent.  Can calculate accuracy for entire epoch.  Set batch size so that it
# can all fit in the GPU
def get_stats_epoch(net, criterion,  data_all, labels_all, batch_size):
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
    return out_accuracy, out_loss


# Produces an ROC curve given softmax activation and labels
def get_roc_curve(softmax, labels):
    y_true = labels
    y_score = softmax
    roc_curve = metrics.roc_curve(y_true, y_score)
    print('roc_curve: ', roc_curve)

