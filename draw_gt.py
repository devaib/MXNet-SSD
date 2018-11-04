import os
import json
import matplotlib.pyplot as plt

dataset_dir = "./data/caltech-pedestrian-dataset-converter"
img_dir = os.path.join(dataset_dir, "data", "images")
annotation_file = os.path.join(dataset_dir, "data", "annotations.json")


class GT:
    def __init__(self, setname, videoname, imgindex):
        self.setname = setname
        self.videoname = videoname
        self.imgindex = imgindex


img_list = [GT("set10", "V010", "930"),
            GT("set10", "V010", "1500"),
            GT("set10", "V010", "720"),
            GT("set09", "V009", "90"),
            GT("set06", "V013", "1800"),
            GT("set06", "V015", "1380"),
            ]

# load json
with open(annotation_file) as f:
    annotations = json.load(f)

    for img in img_list:
        image = annotations[img.setname][img.videoname]['frames'][img.imgindex]
        imagepath = os.path.join(img_dir, img.setname + "_" + img.videoname + "_" + img.imgindex + ".png")

        labels = []
        for detection in image:
            # check label
            cls_name = detection['lbl']
            if not cls_name == 'person':
                continue

            coord = detection['pos']
            xmin = float(coord[0])
            ymin = float(coord[1])
            w = float(coord[2])
            h = float(coord[3])
            labels.append([xmin, ymin, w, h])

        im = plt.imread(imagepath)
        plt.imshow(im)
        color = (0, 0.9, 0)
        for label in labels:
            rect = plt.Rectangle((label[0], label[1]), label[2], label[3], fill=False, edgecolor=color, linewidth=1.5)
            plt.gca().add_patch(rect)

        print(imagepath)
        for label in labels:
            print("{}, {}, {}, {}".format(label[0], label[1], label[2], label[3]))

        plt.show()

