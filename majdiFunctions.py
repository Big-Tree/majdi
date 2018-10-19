import math
import numpy as np

# This function will return the average accuracy for the entire data that it is
# sent.  Can calculate accuracy for entire epoch.  Set batch size so that it
# can all fit in the GPU
def get_accuracy_epoch(net, data_all, labels_all, batch_size):
    print('    length data: ', len(data_all))
    iterations = math.ceil(len(data_all)/batch_size)
    all_accuracies = []
    size_of_final_accuracy = 0
    for i in range(iterations):
        output = net(data_all[i*batch_size : (i+1)*batch_size])
        print('output.shape: ', output.shape)
        print('len(output): ', len(output))
        labels = labels_all[i*batch_size : (i+1)*batch_size]
        print('labels.shape: ', labels.shape)
        maxOutput = np.asarray(
            [np.argmax(_) for _ in output.data.cpu().numpy()])
        print('maxOutput.shape: ', maxOutput.shape)
        pred = np.zeros((len(output), 2))
        print('pred.shape: ', pred.shape)
        pred[range(len(output)), maxOutput] = 1
        print('pred.shape: ', pred.shape)
        accuracy = sum(pred == labels.cpu().numpy())/len(labels)
        print('accuracy.shape: ', accuracy.shape)
        accuracy = accuracy[0]
        print('accuracy.shape: ', accuracy.shape)
        all_accuracies.append(accuracy)
        size_of_final_accuracy = len(output)
        
    out = (sum(all_accuracies[0:-1])/len(all_accuracies)
        + all_accuracies[-1] / len(all_accuracies)
            * size_of_final_accuracy/batch_size)
    sumfirst = sum(all_accuracies[0:-1])
    print('\nsumfirst: ', sumfirst)
    sumfirst = sumfirst / len(all_accuracies)
    print('sumfirst: ', sumfirst)
    part2 = (all_accuracies[-1] / len(all_accuracies)
             * (size_of_final_accuracy/batch_size))
    print('part2: ', part2)
    final = sumfirst + part2
    print('final: ', final)
    print('    out: ', out)
    print('    old: ', sum(all_accuracies)/len(all_accuracies))
    print('    final accuracy: ', all_accuracies[-1])
    print(all_accuracies, '\n')
    return out
