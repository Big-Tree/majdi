import numpy as np
import time
from sklearn import metrics
import torch


# Calculate the y_true and y_score. 
# y_score is the probability of the lesion being present 

# The dataloader should just be the dataloader for the specific dataset
# (not the dictionary of dataloaders
def roc_curve(model, device, dataloader):
    y_score = []
    y_true = []
    y_pred = []
    for dataloader_dict in dataloader:
        inputs = dataloader_dict['image']
        labels = dataloader_dict['label']
        inputs = inputs.to(device, dtype=torch.float)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            tmp = outputs.cpu().numpy()
            y_pred.extend(np.argmax(tmp, axis=1))
            y_score.extend(tmp[:,1]) # Select scores for lesions
            tmp = labels.numpy()
            y_true.extend(tmp[:,1])
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.roc_auc_score(y_true, y_score)
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    return fpr, tpr, auc, sensitivity, specificity
