from torchvision.datasets import MNIST
from torchvision.datasets import EMNIST
from torchvision import transforms
import torch.utils.data.dataloader
import argparse
from torch.utils.data import Dataset
import numpy as np
import random
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns;sns.set()
from scipy.spatial import distance
import matplotlib.pyplot as plt
import torchvision

'''
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
'''
import tensorboardX

parser = argparse.ArgumentParser(description='Simple Contrastive Loss MNIST/EMNIST')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--constractive_loss_margin', type=float, default=0.7)
parser.add_argument('--learning_rate', type=float, default=1e-6)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--validation_split', type=float, default=0.2)
parser.add_argument('--vector_output_size', type=int, default=1024)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--drop_out_ratio', type=float, default=0.5)
parser.add_argument('--dist_metric', type=str, default='euclidean')
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--dataset_train_split', type=float, default=0.6)
parser.add_argument('--data_expansion_factor', type=int, default=1)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--device', type=str, default='cuda')
args, unknown = parser.parse_known_args()

DEVICE = args.device
if not torch.cuda.is_available():
    DEVICE = 'cpu'


class SiameseMNIST(Dataset):

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.data_labels = list(self.data_dict.keys())
        self.data_transforms = transforms.Compose([transforms.ToTensor()])

        positive_pairs = []  # 0
        negative_pairs = []  # 1

        ## datastructure [ img1, img2, target, label1, label2]
        for label in self.data_labels:
            data_for_label = self.data_dict.get(label)
            for i in range(int(len(data_for_label) * args.data_expansion_factor)):
                pairs = random.sample(data_for_label, 2)
                final_pairs = [pairs[0], pairs[1], 0, label, label]
                positive_pairs.append(final_pairs)

        ## datastructure [ img1, img2, target, label1, label2]
        for idx in range(len(positive_pairs)):
            random_labels = random.sample(self.data_labels, 2)
            data_for_label = self.data_dict.get(random_labels[0])
            data_for_non_label = self.data_dict.get(random_labels[1])
            pairs = [random.choice(data_for_label), random.choice(data_for_non_label), 1, random_labels[0],
                     random_labels[1]]
            negative_pairs.append(pairs)

        self.data_pairs = positive_pairs + negative_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        item = self.data_pairs[index]
        img1_t = self.data_transforms(item[0])
        img2_t = self.data_transforms(item[1])
        target = item[2]
        label1 = item[3]
        label2 = item[4]

        return img1_t, img2_t, target, label1, label2

class ContrastiveLoss(nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        if args.dist_metric == 'euclidean':
            distance = F.pairwise_distance(output[0], output[1])
        if args.dist_metric == 'cosine':
            distance = F.cosine_similarity(output[0], output[1])

        loss = 0.5 * (1 - target.float()) * torch.pow(distance, 2) + \
               0.5 * target.float() * torch.pow(torch.clamp(self.margin - distance, min=0.00), 2)

        return loss.mean()


def draw_loss_plot(training_losses, validation_losses, epochs):
    plt.plot(epochs, training_losses, label="train")
    plt.plot(epochs, validation_losses, label="eval")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('training / validation loss')
    plt.legend()
    plt.show()


def draw_accuracy_plot(train_accuracy, test_accuracy, epochs):
    plt.plot(epochs, train_accuracy, label="train")
    plt.plot(epochs, test_accuracy, label="test")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('training / test accuracy')
    plt.legend()
    plt.show()


if args.dataset == 'MNIST':
    train_data = MNIST('../data/MNIST', train=True, download=True)
    test_data = MNIST('../data/MNIST', train=False, download=True)
if args.dataset == 'EMNIST':
    train_data = EMNIST('../data/EMNIST', split='balanced', train=True, download=True)  # 47 balanced classes
    test_data = EMNIST('../data/EMNIST', split='balanced', train=False, download=True)

# split dataset into train and test according to hyperparam 'train_split'
total_classes = list(dict.fromkeys(train_data.train_labels.numpy()))
train_split = int(len(total_classes) * args.dataset_train_split)
labels_to_combine_train_dataset = random.sample(total_classes, k=train_split)
labels_to_combine_test_dataset = list(set(total_classes) - set(labels_to_combine_train_dataset))

print('Train Set Labels : ', labels_to_combine_train_dataset)
print('Test Set Labels : ', labels_to_combine_test_dataset)

combined_train_data = {}
combined_test_data = {}

for label in labels_to_combine_train_dataset:
    train_indexes_1 = np.where(train_data.train_labels == label)[0]
    test_indexes_1 = np.where(test_data.test_labels == label)[0]
    label_list = []
    for index in train_indexes_1:
        label_list.append(train_data[index][0])
    for index in test_indexes_1:
        label_list.append(test_data[index][0])
    combined_train_data[label] = label_list

for label in labels_to_combine_test_dataset:
    train_indexes_2 = np.where(train_data.train_labels == label)[0]
    test_indexes_2 = np.where(test_data.test_labels == label)[0]
    label_list = []
    for index in train_indexes_2:
        label_list.append(train_data[index][0])
    for index in test_indexes_2:
        label_list.append(test_data[index][0])
    combined_test_data[label] = label_list

train_dataset = SiameseMNIST(combined_train_data)
test_dataset = SiameseMNIST(combined_test_data)

split = int(np.floor(args.validation_split * len(train_dataset)))
indices = list(range(len(train_dataset)))
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                           drop_last=True)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

model = torchvision.models.alexnet(pretrained=True)

model.features=nn.Sequential(
    nn.Upsample(scale_factor=8, mode='nearest'), #to match with Alexnet Structure 28px --> 224px
    nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )

model.classifier = nn.Sequential(
                    nn.Dropout(p=args.drop_out_ratio, inplace=False),
                    nn.Linear(in_features=9216, out_features=4096, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=args.drop_out_ratio, inplace=False),
                    nn.Linear(in_features=4096, out_features=4096, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=4096, out_features=args.embedding_size, bias=True)
)

criterion = ContrastiveLoss(margin=args.constractive_loss_margin)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
model = model.to(DEVICE)

train_losses = []
eval_losses = []
epochs = []

train_accuracy = []
test_accuracy = []

best_train_accuracy = 0.0
best_test_accuracy = 0.0

tb_writer = tensorboardX.SummaryWriter()

for epoch in range(1, args.num_epochs + 1):

    epochs.append(epoch)

    ## Training
    for dataloader in [train_loader, val_loader]:

        losses = []

        classes_dict = {}
        projector_labels = []
        projector_imgs = []
        projector_embeddings = []

        if dataloader == train_loader:
            model.train()
        else:
            model.eval()

        for batch in dataloader:

            img1, img2, target, label1, label2 = batch

            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            target = target.to(DEVICE)
            out1 = model(img1)
            out2 = model(img2)
            out1 = out1.to(DEVICE)
            out2 = out2.to(DEVICE)
            out = [out1, out2]
            loss = criterion(out, target)

            if dataloader == train_loader:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

            ##Adding Tensorboard Projector
            labels_total = torch.cat((label1, label2), dim=0)
            imgs_total = torch.cat((img1, img2), dim=0)
            outs_total = torch.cat((out1, out2), dim=0)

            for label, img, out in zip(labels_total, imgs_total, outs_total):
                label = label.item()
                if label not in list(classes_dict.keys()):
                    classes_dict[label] = 1
                    projector_labels.append(label)
                    projector_embeddings.append(out.detach().cpu())
                    projector_imgs.append(img.detach().cpu())
                else:
                    current_count = classes_dict.get(label)
                    if current_count < args.no_projector_samples_per_class:
                        classes_dict[label] = current_count + 1
                        projector_labels.append(label)
                        projector_embeddings.append(out.detach().cpu())
                        projector_imgs.append(img.detach().cpu())

        if dataloader == train_loader:
            train_losses.append(np.mean(losses))
            print('epoch', epoch, 'train_loss', np.mean(losses))
            tb_writer.add_scalars(tag_scalar_dict={'Train': np.mean(losses)}, global_step=epoch,
                                  main_tag='Loss')
            tb_writer.add_embedding(
                mat=torch.FloatTensor(np.stack(projector_embeddings)),
                label_img=torch.FloatTensor(np.stack(projector_imgs)),
                metadata=projector_labels,
                global_step=epoch, tag=f'train_emb_{epoch}')
            tb_writer.flush()
        else:
            eval_losses.append(np.mean(losses))
            print('epoch', epoch, 'val_loss', np.mean(losses))
            tb_writer.add_scalars(tag_scalar_dict={'Eval': np.mean(losses)}, global_step=epoch,
                                  main_tag='Loss')
            tb_writer.add_embedding(
                mat=torch.FloatTensor(np.stack(projector_embeddings)),
                label_img=torch.FloatTensor(np.stack(projector_imgs)),
                metadata=projector_labels,
                global_step=epoch, tag=f'val_emb_{epoch}')
            tb_writer.flush()

    #if len(train_accuracy) > 1 and (train_accuracy[epoch - 1] < train_accuracy[epoch]):
        #torch.save(model, 'best.pth')
        #best_train_accuracy = train_accuracy[epoch]
    #if len(test_accuracy) > 1 and (test_accuracy[epoch - 1] < test_accuracy[epoch]):
        #best_test_accuracy = test_accuracy[epoch]
    torch.save(model, 'last.pth')

    model = torch.load('last.pth')
    model = model.to(DEVICE)
    model.eval()

    ##Testing
    for dataloader in [train_loader, test_loader]:

        embbedding_dict = {}
        embbedding_center_dict = {}

        ##center calculation
        for batch in dataloader:
            img1, img2, target, label1, label2 = batch

            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)

            out1 = model(img1)
            out2 = model(img2)

            ##collecting all embedding for each label
            for index, label in enumerate(label1):
                if label.item() in embbedding_dict.keys():
                    existing_emb_for_label = embbedding_dict.get(label.item())
                    existing_emb_for_label.append(out1[index].data[:])
                else:
                    embbedding_dict[label.item()] = [out1[index].data[:]]
            for index, label in enumerate(label2):
                if label.item() in embbedding_dict.keys():
                    existing_emb_for_label = embbedding_dict.get(label.item())
                    existing_emb_for_label.append(out2[index].data[:])
                else:
                    embbedding_dict[label.item()] = [out2[index].data[:]]

        ##calculate center for each label
        for label in embbedding_dict.keys():
            embeddings = embbedding_dict.get(label)
            embbedding_center_dict[label] = torch.mean(torch.stack(embeddings))

        batch_accuracy = []

        ##accuracy calculation
        for batch in dataloader:
            img1, img2, target, label1, label2 = batch

            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)

            out1 = model(img1)
            out2 = model(img2)

            accuracy = 0

            for s_out1, s_out2, s_target, s_label1, s_label2 in zip(out1, out2, target, label1, label2):

                center_labels = np.array(list(embbedding_center_dict.keys()))
                center_values = [embbedding_center_dict.get(label) for label in center_labels]

                out1_distances = [abs(center - torch.mean(s_out1.data[:])) for center in center_values]
                out1_closest_label = center_labels[out1_distances.index(min(out1_distances))]

                out2_distances = [abs(center - torch.mean(s_out2.data[:])) for center in center_values]
                out2_closest_label = center_labels[out2_distances.index(min(out2_distances))]

                if (((out1_closest_label == out2_closest_label) and s_target == 0) or
                        ((out1_closest_label != out2_closest_label) and s_target == 1)):
                    accuracy = accuracy + 1
            batch_accuracy.append(accuracy/args.batch_size)

            ##Adding Tensorboard Projector
            labels_total = torch.cat((label1, label2), dim=0)
            imgs_total = torch.cat((img1, img2), dim=0)
            outs_total = torch.cat((out1, out2), dim=0)

            for label, img, out in zip(labels_total, imgs_total, outs_total):
                label = label.item()
                if label not in list(classes_dict.keys()):
                    classes_dict[label] = 1
                    projector_labels.append(label)
                    projector_embeddings.append(out.detach().cpu())
                    projector_imgs.append(img.detach().cpu())
                else:
                    current_count = classes_dict.get(label)
                    if current_count < args.no_projector_samples_per_class:
                        classes_dict[label] = current_count + 1
                        projector_labels.append(label)
                        projector_embeddings.append(out.detach().cpu())
                        projector_imgs.append(img.detach().cpu())

        # epoch accuracy
        if dataloader == train_loader:
            print('train epoch acc', epoch, np.mean(batch_accuracy))
            train_accuracy.append(np.mean(batch_accuracy) * 100)
            tb_writer.add_scalars(tag_scalar_dict={'Train': (np.mean(batch_accuracy) * 100)}, global_step=epoch,
                                  main_tag='Accuracy')
        elif dataloader == test_loader:
            print('test epoch acc', epoch, np.mean(batch_accuracy))
            test_accuracy.append(np.mean(batch_accuracy) * 100)
            tb_writer.add_scalars(tag_scalar_dict={'Test': (np.mean(batch_accuracy) * 100)}, global_step=epoch,
                                  main_tag='Accuracy')
            tb_writer.add_embedding(
                mat=torch.FloatTensor(np.stack(projector_embeddings)),
                label_img=torch.FloatTensor(np.stack(projector_imgs)),
                metadata=projector_labels,
                global_step=epoch, tag=f'test_emb_{epoch}')
            tb_writer.flush()

draw_loss_plot(train_losses, eval_losses, epochs)
draw_accuracy_plot(train_accuracy, test_accuracy, epochs)


