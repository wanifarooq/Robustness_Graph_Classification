import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import csv
from utils import set_seed
import math

set_seed()

folder_path = "distribution"
csv_file_path = 'variable_values.csv'
model_dir = 'model'
x_min_fixed_value = 0
x_max_fixed_value = 0
y_min_fixed_value = 0
y_max_fixed_value = 0.8

def makedistdir(dataset_name):
    global folder_path, csv_file_path, model_dir
    folder_path = folder_path + "_" + dataset_name
    model_dir = model_dir + "_" + dataset_name
    csv_file_path = csv_file_path + "_" + dataset_name
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)
    header = ['epoch', 'value', 'class']
    drop_class = []
    small_class = []
    # Write the header to the CSV file (if the file doesn't exist)
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
    return model_dir

def distribution(classes_distance, distanceFromall_list, epoch, types, class2vis):
    global x_min_fixed_value, x_max_fixed_value, y_min_fixed_value, y_max_fixed_value
    if epoch == 0:
        for i in range(0, len(distanceFromall_list)):
            max_val = math.ceil(3 * distanceFromall_list[i].max().item())
            if max_val > x_max_fixed_value:
                x_max_fixed_value = max_val
    num_bins = 20

    # dip_classes = [diptest.diptest(vectors_np.cpu().numpy()) for vectors_np in classes_distance]
    # dip_distance_from_all_list = [diptest.diptest(distances.cpu().numpy()) for distances in distanceFromall_list]

    histograms_classes = [np.histogram(vectors_np.cpu().numpy(), bins=num_bins, density=False) for vectors_np in
                          classes_distance]
    histograms_distance_from_all_list = [np.histogram(distances.cpu().numpy(), bins=num_bins, density=False)
                                          for distances in distanceFromall_list]

    histograms_torch_classes = [torch.tensor(hist[0], dtype=torch.float32) for hist in histograms_classes]
    histograms_torch_distance_from_all_list = [torch.tensor(hist[0], dtype=torch.float32)
                                               for hist in histograms_distance_from_all_list]

    bin_edges_torch_classes = [torch.tensor(hist[1], dtype=torch.float32) for hist in histograms_classes]
    bin_edges_torch_distance_from_all_list = [torch.tensor(hist[1], dtype=torch.float32)
                                              for hist in histograms_distance_from_all_list]

    # Calculate the continuous density estimate for each dimension using the midpoint of each bin
    x_values_classes = [0.5 * (bin_edges[:-1] + bin_edges[1:]) for bin_edges in bin_edges_torch_classes]
    x_values_distance_from_all_list = [0.5 * (bin_edges[:-1] + bin_edges[1:])
                                        for bin_edges in bin_edges_torch_distance_from_all_list]
    # totalClassdensity = [torch.sum(x) + torch.sum(y) for x, y in
    #                      zip(histograms_torch_classes, histograms_torch_distance_from_all_list)]
    density_estimates_classes = [hist / torch.sum(hist) for hist in histograms_torch_classes]
    density_estimates_distance_from_all_list = [hist / torch.sum(hist)
                                                 for hist in  histograms_torch_distance_from_all_list]

    colors = ['purple', 'gold', 'teal', 'pink', 'lime', 'gray', 'navy','red']
    cmap = mcolors.ListedColormap(colors)

    def save_to_csv(file_path, data):
        with open(file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(data)

    for i in range(class2vis):
        plt.plot(x_values_classes[i], density_estimates_classes[i], color=cmap(i), label=types.split('-')[0]+f'_{i}', linewidth=2.5)

    # Plot the distance from all for each class
    for i in range(class2vis):
        plt.plot(x_values_distance_from_all_list[i], density_estimates_distance_from_all_list[i],
                 color=cmap(i + class2vis), label=types.split('-')[1]+f'_{i}', linewidth=2.5)

    plt.xlim(x_min_fixed_value, x_max_fixed_value)
    plt.ylim(y_min_fixed_value, y_max_fixed_value)
    plt.xlabel('Distance from Centroid')
    plt.ylabel('Density Of Samples')
    plt.legend()

    path = os.path.join(folder_path, types + str(epoch) + ".png")
    plt.savefig(path)
    plt.close()
    return None




# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import os
# import diptest
# import csv
# from utils import set_seed
# import math
# set_seed()
#
# folder_path = "distribution"
# csv_file_path = 'variable_values.csv'
# model_dir = 'model'
# x_min_fixed_value = 0
# x_max_fixed_value = 0
# y_min_fixed_value = 0
# y_max_fixed_value = 0.6
# def makedistdir(dataset_name):
#     global folder_path,csv_file_path,model_dir
#     folder_path = folder_path+"_"+dataset_name
#     model_dir = model_dir+"_"+dataset_name
#     csv_file_path = csv_file_path+"_"+dataset_name
#     os.makedirs(model_dir,exist_ok=True)
#     os.makedirs(folder_path, exist_ok=True)
#     header = ['epoch', 'value', 'class']
#     drop_class = []
#     small_class = []
#     # Write the header to the CSV file (if the file doesn't exist)
#     with open(csv_file_path, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(header)
#     return model_dir
# def distribution(classes_distance,distanceFromall_list,epoch,types,class2vis):
#     global x_min_fixed_value,x_max_fixed_value,y_min_fixed_value,y_max_fixed_value
#     if epoch == 0:
#         for i in range(0,len(distanceFromall_list)):
#             max = math.ceil(2 * distanceFromall_list[i].max().item())
#             if max> x_max_fixed_value:
#                 x_max_fixed_value = max
#     num_bins = 20
#
#     dip = [diptest.diptest(vectors_np.cpu().numpy()) for vectors_np in classes_distance]
#
#     histogram = [np.histogram(vectors_np.cpu().numpy(), bins=num_bins, density=True) for vectors_np in classes_distance]
#
#     histograms_torch = [torch.tensor(hist[0], dtype=torch.float32) for hist in histogram]
#     bin_edges_torch =  [torch.tensor(hist[1], dtype=torch.float32) for hist in histogram]
#
#     # Calculate the continuous density estimate for each dimension using the midpoint of each bin
#     x_values = [0.5 * (bin_edges[:-1] + bin_edges[1:]) for bin_edges in bin_edges_torch]
#     density_estimates = [hist / torch.sum(hist) for hist in histograms_torch]
#     # colors = ['red', 'blue', 'green', 'purple']
#     colors = ['purple', 'gold', 'teal', 'pink', 'lime', 'gray', 'navy']
#     cmap = mcolors.ListedColormap(colors)
#
#     def save_to_csv(file_path, data):
#         with open(file_path, 'a', newline='') as csvfile:
#             csv_writer = csv.writer(csvfile)
#             csv_writer.writerow(data)
#
#
#
#     # values = ([t[1] for t in dip])
#     # indexlist = [index for index, value in enumerate(values) if value <= 0.05 and index not in drop_class]
#     # if indexlist:
#     #     for i in indexlist:
#     #         mini =  values[i]
#     #         epoch_data = [epoch, f'{mini:.3f}', i]
#     #         save_to_csv(csv_file_path, epoch_data)
#     #         plt.plot(x_values[i], density_estimates[i], label=f'class {i}', linewidth=2.5)
#     #         drop_class.append(i)
#     #         small_class.append(i)
#     # else:
#     #     mini = min(values)
#     #     indexi = values.index(mini)
#     #     values_copy = values[:]
#     #     while indexi in small_class:
#     #         values_copy.remove(mini)
#     #         mini = min(values_copy)
#     #         indexi = values.index(mini)
#     #     small_class.append(indexi)
#     #     epoch_data = [epoch, f'{mini:.3f}', indexi]
#     #     save_to_csv(csv_file_path, epoch_data)
#     #
#     #     plt.plot(x_values[indexi], density_estimates[indexi], label=f'class_smaller {indexi}', linewidth=2.5)
#     for i in range(class2vis):
#         # plt.plot(x_values[i], density_estimates[i],color=cmap(i) ,label=f'class {dip[i][1]}',linewidth=2.5)
#         # plt.plot(x_values[i], density_estimates[i], label=f'class {dip[i][1]}', linewidth=2.5)
#         plt.plot(x_values[i], density_estimates[i],color=cmap(i) , label=f'class {i}', linewidth=2.5)
#         # print()
#     plt.xlim(x_min_fixed_value, x_max_fixed_value)
#     plt.ylim(y_min_fixed_value, y_max_fixed_value)
#     plt.xlabel('Distance from Centroid')
#     plt.ylabel('Density Of Samples')
#     plt.legend()
#
#
#     path = os.path.join(folder_path, types + str(epoch) + ".png")
#     plt.savefig(path)
#     plt.close()
#     return None
