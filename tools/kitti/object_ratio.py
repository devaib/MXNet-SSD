import os
import csv

valid_cls = ['Car', 'Van', 'Truck']
to_file = '../../data/kitti/results/imginfos_valset.txt'
label_path = '../../data/kitti/data_object_label_2/training/label_2'
small_object_label_path = '../../data/kitti/data_object_label_2/training/label_small_objects'
validation_set_only = True  # only genrate info for validation set
if validation_set_only:
    val_file = '../../data/kitti/data_object_image_2/training/val.txt'
    with open(val_file) as f:
        val_set = f.read().splitlines()

imginfos = []
for root, dirs, files in os.walk(label_path):
    for filename in files:
        filepath = os.path.join(label_path, filename)
        index = filename.split('.')[0]
        if validation_set_only and (index not in val_set):
            continue

        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                cls_name = words[0]
                if cls_name not in valid_cls:
                    continue
                x = int(round(float(words[4])))
                y = int(round(float(words[5])))
                w = int(round(float(words[6]) - float(words[4])))
                h = int(round(float(words[7]) - float(words[5])))
                ratio = (float(words[7])-float(words[5])) / (float(words[6])-float(words[4]))
                if ratio > 5.0:  # skip bad labels
                    continue
                imgsize = w * h
                imginfo = [index, x, y, w, h, ratio, imgsize]
                imginfos.append(imginfo)

# write to file
with open(to_file, 'w+') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(imginfos)

