import numpy as np
import cv2
import os


folder = "/scratch/mjafarzadeh/results_2/H_5"
output_folder = "/scratch/mjafarzadeh/images_2/BH_5/"

file_list = []
for file in os.listdir(folder):
    if "_new_clusters_" in file:
        file_list.append(os.path.join(folder, file))
file_list.sort()

my_dict = dict()
N_cluster = 0

for file in file_list:
    with open(file, "r") as f:
        for L in f:
            line = L.strip("\n")
            if len(line) > 4:
                if "cluster" in line:
                    cluster = int(line.split()[1])
                    my_dict[cluster] = list()
                    N_cluster = N_cluster + 1
                elif "ImageNet" in line:
                    my_dict[cluster].append("/scratch/datasets/" + line)
                else:
                    raise ValueError()


for cluster in range(1, 1 + N_cluster):
    print(f"{cluster = }")
    images_list = my_dict[cluster]

    n = int(np.ceil(np.sqrt(len(images_list))))
    m = int(np.ceil(len(images_list) / n))
    N = len(images_list)

    s = 0
    for x in images_list:
        if "ImageNet/v2_" in x:
            s = s + 1

    if s < (N / 5):
        prefix = "good_"
    elif s >= (N / 2):
        prefix = "bad_"
    else:
        prefix = "rest_"

    print(f"{( m , n , m*n , N ) = }")

    output = np.zeros((320 * m, 320 * n, 3))

    for i in range(m):
        for j in range(n):
            k = (i * m) + j
            if k < N:
                image_filename = images_list[k]
                A = image_filename
                img = cv2.imread(A, 1)
                resized = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)
                i1 = 320 * i
                i2 = 320 * (i + 1)
                j1 = 320 * j
                j2 = 320 * (j + 1)
                output[i1:i2, j1:j2, :] = resized

    cv2.imwrite(output_folder + prefix + str(cluster).zfill(2) + ".jpg", output)
