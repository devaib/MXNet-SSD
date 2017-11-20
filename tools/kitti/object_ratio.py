import os
import csv

valid_cls = ['Car', 'Van', 'Truck']
label_path = '../../data/kitti/data_object_label_2/training/label_2'
small_object_label_path = '../../data/kitti/data_object_label_2/training/label_small_objects'

imginfos = []
for root, dirs, files in os.walk(label_path):
    for filename in files:
        filepath = os.path.join(label_path, filename)
        index = filename.split('.')[0]

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
to_file = '../../data/kitti/results/imginfos.txt'
with open(to_file, 'w+') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(imginfos)

