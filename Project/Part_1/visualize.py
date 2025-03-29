import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from load_dataset import dataset

def visualize_sample(image, labels):
    ax = plt.subplots(1)
    ax.imshow(image)
    
    for obj in labels:
        rect = patches.Rectangle(
            (obj["Left"], obj["Top"]),
            obj["Right"] - obj["Left"],
            obj["Bottom"] - obj["Top"],
            linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

random_indices = random.sample(range(len(dataset)), 2)

sample_image_1, sample_labels_1 = dataset[random_indices[0]]
visualize_sample(sample_image_1, sample_labels_1)

sample_image_2, sample_labels_2 = dataset[random_indices[1]]
visualize_sample(sample_image_2, sample_labels_2)