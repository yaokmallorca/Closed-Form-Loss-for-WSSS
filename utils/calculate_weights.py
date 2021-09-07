import os
from tqdm import tqdm
import numpy as np
# from mypath import Path

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/media/data/VOCdevkit/VOCSUB/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/media/data/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/media/data/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/media/data/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        # print("sample: ", len(sample), sample[3].size())
        y = sample[3] # sample['label']
        y = y.detach().cpu().numpy()
        mask = (y > 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    np.save(classes_weights_path, ret)
    return ret