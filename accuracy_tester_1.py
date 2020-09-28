import torch
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np


num_of_samples = 10000
X, y, centers = make_blobs(n_samples=num_of_samples, centers=10, n_features=128, random_state=1, shuffle=True,
                           cluster_std=0.3, return_centers=True)

color_dict = {}
is_show = True


def create_colors(no_colors = 10):
    for i in range(no_colors):
        color = list(np.random.choice(range(256), size=3))
        color = "#{0:02x}{1:02x}{2:02x}".format(color[0], color[1], color[2])
        color_keys = list(color_dict.keys())
        if len(color_keys) > 0:
            for key in color_keys:
                if color == color_dict.get(key):
                    continue
                color_dict[i] = color
        else:
            color_dict[i] = color

    return color_dict


if is_show:
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    colors = create_colors(10)
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()


embbedding_dict = {}
##collecting all embedding for each label
for index, label1 in enumerate(y):
    if label1 in embbedding_dict.keys():
        existing_emb_for_label = embbedding_dict.get(label1)
        existing_emb_for_label.append(torch.from_numpy(X[index]))
        embbedding_dict[label1] = existing_emb_for_label
    else:
        embbedding_dict[label1] = [torch.from_numpy(X[index])]


embbedding_center_dict = {}
##calculate center for each label
for label in embbedding_dict.keys():
    embeddings = embbedding_dict.get(label)
    embbedding_center_dict[label] = torch.mean(torch.stack(embeddings), dim=0)

keys_ascending = list(embbedding_center_dict.keys())
keys_ascending.sort()

all_centers_ascending = [embbedding_center_dict[label] for label in keys_ascending]
print(len(all_centers_ascending))
accuracy = 0

for x, y1 in zip(X, y):
    dists = torch.pairwise_distance(torch.stack(all_centers_ascending), torch.stack(len(all_centers_ascending)*[torch.from_numpy(x)]), p=2)
    closest_idx = torch.argmin(dists).item()
    if closest_idx == y1:
        accuracy += 1
print('Accuracy : ', accuracy/len(X)*100)

