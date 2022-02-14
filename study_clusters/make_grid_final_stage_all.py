import numpy as np
import torch
import cv2
import os


folder = "/scratch/mjafarzadeh/result_5_svm/"
output_root = "/scratch/mjafarzadeh/grid_5_svm/"

for level in ["easy", "hard"]:
    for test_id in range(1, 6):
        print(f"test {level} {test_id}")

        output_folder = output_root + f"{level}_{test_id}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_folder = output_folder + "/"

        data = torch.load(folder + f"track_svm_{level}_{test_id}_final_stage.pth")
        keys = list(data.keys())

        set_all_classes = set()
        for x in keys:
            number, tp = x.split("_")
            set_all_classes.add(int(number))
        list_all_classes = sorted(list(set_all_classes))
        N = list_all_classes[-1]

        for class_number in list_all_classes:
            if f"{class_number}_rest" in keys:
                images_rest = sorted(data[f"{class_number}_rest"])
            else:
                images_rest = []
            if f"{class_number}_rejected" in keys:
                images_rejected = sorted(data[f"{class_number}_rejected"])
            else:
                images_rejected = []

            images_list = images_rest + ["Null"] + images_rejected

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
                        i1 = 320 * i
                        i2 = 320 * (i + 1)
                        j1 = 320 * j
                        j2 = 320 * (j + 1)
                        if image_filename != "Null":
                            A = "/scratch/datasets/" + image_filename
                            img = cv2.imread(A, 1)
                            resized = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)
                            output[i1:i2, j1:j2, :] = resized
                        else:
                            output[i1:i2, j1:j2, 2] = 255

            cv2.imwrite(output_folder + "class_" + str(class_number).zfill(4) + ".jpg", output)
