import math
import numpy as np
from sklearn import metrics


# pass the whole dataset and the minibatch size and then use the functions to
# get minibatches
class Minibatch:
    def __init__(self, data, labels, minibatch_size):
        self.data = data
        self.labels = labels
        self.minibatch_size = minibatch_size
        self.num_iterations = math.ceil(len(data)/minibatch_size)
        self.i = -1 # count of the number of iterations passed

    def get_data(self):
        self.i += 1
        out = self.data[self.i*self.minibatch_size :
                         (self.i+1)*self.minibatch_size]
        return out

    def get_labels(self):
        out = self.labels[self.i*self.minibatch_size :
                          (self.i+1)*self.minibatch_size]
        return out

# This function will return the average accuracy for the entire data that it is
# sent.  Can calculate accuracy for entire epoch.  Set batch size so that it
# can all fit in the GPU
def get_stats_epoch(net, criterion,  data_all, labels_all, batch_size):
    minibatch = Minibatch(data_all, labels_all, batch_size)
    all_accuracies = []
    all_losses = []
    size_of_final_accuracy = 0
    for i in range(minibatch.num_iterations):
        output = net(minibatch.get_data())
        labels = minibatch.get_labels()
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
# Will be given net, criterion, data_all, labels_all, batch_size
def get_roc_curve(net, data_all, labels_all, optimizer, batch_size):
    minibatch = Minibatch(data_all, labels_all, batch_size)
    print('minibatch.data.shape: ', minibatch.data.shape)
    print('minibatch.labels.shape: ', minibatch.labels.shape)
    print('minibatch.minibatch_size: ', minibatch.minibatch_size)
    print('minibatch.num_iterations: ', minibatch.num_iterations)
    y_score = []
    y_true = labels_all
    for i in range(minibatch.num_iterations):
        optimizer.zero_grad()
        print('i: ', i)
        net.save_softmax = True
        out_data = minibatch.get_data()
        #print('out_data.shape: ',out_data.shape)
        output = net(out_data)
        y_score.extend(net.softmax_out.detach().cpu().numpy())
        print(net.softmax_out.detach().cpu().numpy())
        #print('y_score.shape: ', np.asarray(y_score).shape)
        #print('y_true.shape: ', np.asarray(y_true).shape)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    print('y_true.shape: ', y_true.shape)
    print('y_score.shape: ', y_score.shape)
    # Unconvert from onehot format
    y_true = y_true[:,1]
    print('y_true.shape: ', y_true.shape)
    print('y_score.shape: ', y_score.shape)
    print('y_true: ', y_true)
    print('y_score: ', y_score)
    y_true = y_true.astype(int)
    y_score = y_score[range(len(y_score)), y_true]
    print('y_true.shape: ', y_true.shape)
    print('y_score.shape: ', y_score.shape)
    print('y_true: ', y_true)
    print('y_score: ', y_score)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.roc_auc_score(y_true, y_score)
    return fpr, tpr, auc
