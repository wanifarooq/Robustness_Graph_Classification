import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
# from PIL import Image
import os
import torch
import matplotlib.colors as mcolors
from utils import set_seed
from matplotlib.lines import Line2D




set_seed()
folder_path = "images"
def makeImagedir(dataset_name):
    global folder_path
    folder_path = folder_path+"_"+dataset_name
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path+"/grad", exist_ok=True)
def visualization(vector,epoch,seed_emd,class_seed_flag,flag,class_to_visualize):
    cut=int(len(vector)/(2*(len(class_seed_flag))))
    # class_seed_flag = class_seed_flag.float()
    vector = vector.view(len(vector), vector.size(1))
    seed_emd = seed_emd.view(len(seed_emd),seed_emd.size(1))
    vector = torch.cat((vector, seed_emd), dim=0)
    tsne = TSNE(n_components=3, perplexity=30, learning_rate=100, n_iter=2000)
    embedded_data = tsne.fit_transform(vector.cpu().numpy())


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define markers and colors for each class
    # markers = ['o', 's', '^', 'v', 'd', 'p', '*', 'h', 'x', '8']
    markers = ['o', 's', '^', 'v', 'd', 'p', '*', 'h', 'x', '+', '>', '<', 'D', '1']
    colors = ['red', 'blue', 'green','purple']
    # colors = ['purple', 'gold', 'teal', 'pink', 'lime', 'gray', 'navy']
    cmap = mcolors.ListedColormap(colors)
    # Plot data points with class labels
    if flag:
        for i in range(class_to_visualize):
            k= 2*cut*i
            class_data = embedded_data[k:(k+cut)]
            ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], marker=markers[i], color=cmap(i),alpha=0.8, s=150,
                       label=f'Class {i}')
            class_data = embedded_data[(k+cut):(k + (2*cut))]
            ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], marker=markers[i+4], color=cmap(i),alpha=0.8, s=150,
                       label=f'Class {i}')
    else:
        for i in range(class_to_visualize):
            k = 2 * cut * i
            class_data = embedded_data[k:(k +( 2*cut))]
            ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], marker=markers[i], color=cmap(i), alpha=0.8,
                       s=150,label=f'Class {i}')

    seed_emd=embedded_data[2*cut*len(class_seed_flag):]
    for x,i in zip(seed_emd,range(class_to_visualize)):
        if class_seed_flag[i]:
            mark =i
        else:
            mark = i+4
        ax.scatter(x[0], x[1], x[2], marker=markers[mark], color=cmap(i), alpha=0.8,
                   s=300,
                   label=f'Class {i}')
    ax.set_title("t-SNE 3D Visualization with Class Labels")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend().remove()
    # plt.show()
    image_path = os.path.join(folder_path, "tsne_visualization"+str(epoch)+".png")
    plt.savefig(image_path)
    plt.close(fig)

    return None


def plot_grad_u(ave_grads,max_grads,epoch):
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    # plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    path = os.path.join(folder_path, "grad/grad_u_visualization" + str(epoch) + ".png")
    plt.savefig(path)
    plt.close()
def plot_grad_flow(named_parameters,epoch):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n.split(".")[-2])
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    path = os.path.join(folder_path, "grad/grad_net_visualization" + str(epoch) + ".png")
    plt.savefig(path)
    plt.close()