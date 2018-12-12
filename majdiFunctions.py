import math
import numpy as np
from sklearn import metrics
from scipy import stats as sc_stats
import os

# Save the training stats as a text file to make it easy to repeat the
# Writes the results to file
# prints results to terminal
def save_results(directory, stats, num_runs):
    if directory != None:
        if os.path.exists(directory) == False:
            os.makedirs(directory)

        # print to file
        with open(directory + '/results.txt', 'w') as f:
            print('Averaging of {:.0f} runs'.format(num_runs), file=f)
            for phase in stats[0]:
                print('{}:'.format(phase), file=f)
                for metric in stats[0][phase]:
                    average = np.array([])
                    for i in range(len(stats)):
                        average = np.append(average, stats[i][phase][metric])
                    # print average and standard error of the mean
                    print('    {}: {:.3f} ({:.4f})'.format(
                        metric, np.average(average), sc_stats.sem(average)), file=f)
            # Dump the raw results
            f.write('\n\n\nData dump:\n')
            for phase in stats[0]:
                f.write('\n' + phase + ':')
                for metric in stats[0][phase]:
                    f.write('\n  ' + metric + ':\n    ')
                    for i in range(len(stats)):
                        f.write(str(stats[i][phase][metric]) + ', ')

   # # print to terminal
   # print('\nAveraging of {:.0f} runs'.format(num_runs))
   # for phase in stats[0]:
   #     for metric in stats[0][phase]:
   #         average = 0
   #         for i in range(len(stats)):
   #             average += stats[i][phase][metric]
   #         print('    {}: {:.3f}'.format(
   #             metric, average/len(stats)))


    # print to terminal
    print('\nAveraging of {:.0f} runs'.format(num_runs))
    for phase in stats[0]:
        print('{}:'.format(phase))
        for metric in stats[0][phase]:
            average = np.array([])
            for i in range(len(stats)):
                average = np.append(average, stats[i][phase][metric])
            # print average and standard error of the mean
            print('    {}: {:.3f} ({:.4f})'.format(
                metric, np.average(average), sc_stats.sem(average)))


# experiment
def save_training_stats(directory):
    text_file = open('training_stats.txt', 'w')
    text_file.write('something something')
    text_file.close()


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
    for i in range(minibatch.num_iterations):
        output = net(minibatch.get_data())
        labels = minibatch.get_labels()
        maxOutput = np.asarray(
            [np.argmax(_) for _ in output.data.cpu().numpy()])
        pred = np.zeros((len(output), 2))
        pred[range(len(output)), maxOutput] = 1
        accuracy = sum(pred == labels.cpu().numpy())/len(labels)
        loss = criterion(output, labels)
        loss = loss.item()
        all_losses.append(loss*(len(output)/len(minibatch.data)))
        accuracy = accuracy[0]
        all_accuracies.append(accuracy*(len(output)/len(minibatch.data)))

    out_accuracy = sum(all_accuracies)
    out_loss = sum(all_losses)

    return out_accuracy, out_loss

# Produces an ROC curve given softmax activation and labels
# Will be given net, criterion, data_all, labels_all, batch_size
def get_roc_curve(net, data_all, labels_all, optimizer, batch_size):
    minibatch = Minibatch(data_all, labels_all, batch_size)
    y_score = []
    y_true = labels_all
    for i in range(minibatch.num_iterations):
        optimizer.zero_grad()
        net.save_softmax = True
        out_data = minibatch.get_data()
        output = net(out_data)
        y_score.extend(net.softmax_out.detach().cpu().numpy())
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Unconvert from onehot format
    y_true = y_true[:,1] 

    y_true = y_true.astype(int)
    # y_score is the probability of a lesion being present
    y_score = y_score[range(len(y_score)), 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.roc_auc_score(y_true, y_score)
    return fpr, tpr, auc


