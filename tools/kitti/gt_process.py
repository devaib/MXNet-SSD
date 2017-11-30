import os
import csv

# parse gt for validation set
label_path = '../../data/kitti/data_object_label_2/training/label_2'
val_path = '../../data/kitti/data_object_image_2/training/val.txt'
valid_cls = ['Car', 'Van', 'Truck']
with open(val_path) as f:
    imgnames = [idx.rstrip() for idx in f.readlines()]

gts = []
for imgname in imgnames:
    label_file = os.path.join(label_path, imgname + '.txt')
    assert os.path.exists(label_file), "Path does not exist {}".format(label_file)
    with open(label_file) as f:
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
            gt = [imgname, x, y, w, h]
            gts.append(gt)

# write to file
to_file = '../../data/kitti/results/gts.txt'
with open(to_file, 'w+') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(gts)



