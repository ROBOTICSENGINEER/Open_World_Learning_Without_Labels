import numpy as np
import cv2
import os


folder = "/scratch/mjafarzadeh/result_5_svm/"
output_root = "/scratch/mjafarzadeh/grid_5_svm/"


for level in ["easy", "hard"]:
    for test_id in range(1, 6):
        print(f"test {level} {test_id}")
        output_folder = output_root + f"{level}_{test_id}/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_list = []
        for file in os.listdir(folder):
            if f"track_svm_{level}_{test_id}_new_clusters_" in file:
                file_list.append(os.path.join(folder, file))
        file_list.sort()

        cluster_image_address_dict = dict()
        N_cluster = 0

        for file in file_list:
            with open(file, "r") as f:
                for L in f:
                    line = L.strip("\n")
                    if len(line) > 4:
                        if "cluster" in line:
                            cluster = int(line.split()[1])
                            cluster_image_address_dict[cluster] = list()
                            N_cluster = N_cluster + 1
                        elif "ImageNet" in line:
                            cluster_image_address_dict[cluster].append("/scratch/datasets/" + line)
                        else:
                            raise ValueError()

        for cluster in cluster_image_address_dict.keys():
            im_list = cluster_image_address_dict[cluster]
            images_list = sorted(im_list)

            print(f"{len(images_list) = }")

            n = int(np.ceil(np.sqrt(len(images_list))))
            m = int(np.ceil(len(images_list) / n))
            N = len(images_list)

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

            cv2.imwrite(output_folder + "cluster_" + str(cluster).zfill(2) + ".jpg", output)
