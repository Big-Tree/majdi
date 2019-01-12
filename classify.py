import torch
import numpy as np

# Returns a dictionary with predicted class and softmax output
# Class is just the argmax of the softmax
# label is the actual ground truth
def classify_images(model, dataloader, device):
    print('classifying images')
    model.eval()
    out = {}
    for dataloader_dict in dataloader['test']:
        inputs = dataloader_dict['image']
        labels = dataloader_dict['label']
        labels = labels.to('cpu').numpy()
        file_name = dataloader_dict['file_name']
        inputs = inputs.to(device, dtype=torch.float)
        with torch.no_grad():
            # Create a dictionary with file_name as key and classification
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            preds = preds.to('cpu').numpy()
            soft = outputs.to('cpu').numpy()
            for name, classification, soft_, label_ in zip(file_name,
                                                  preds,
                                                  soft,
                                                  labels):
                out[name] = {'class': classification,
                             'soft': soft_,
                             'label': label_}
    return out


