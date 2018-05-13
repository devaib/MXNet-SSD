import os
import csv
import json

valid_cls = ['person']
to_file = '../../data/caltech-pedestrian-dataset-converter/results/imginfos.txt'
label_path = '../../data/caltech-pedestrian-dataset-converter/data/annotations.json'
train_set = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']
val_set = ['set06', 'set07', 'set08', 'set09', 'set10']

imginfos = []
with open(label_path) as f:
    annos = json.load(f)
    labels = dict()
    people_list = []

    for setname in annos.keys():
        #if setname not in val_set:
        #    continue
        for videoname in annos[setname].keys():
            print setname, videoname
            for imagename in annos[setname][videoname]['frames'].keys():
                image = annos[setname][videoname]['frames'][imagename]
                for detection in image:
                    cls_name = detection['lbl']
                    if cls_name not in valid_cls:
                        continue
                    coord = detection['pos']
                    x = coord[0]
                    y = coord[1]
                    w = coord[2]
                    h = coord[3]
                    ratio = float(w) / float(h)
                    imginfo = [setname, videoname, imagename,
                               x, y, w, h, ratio]
                    imginfos.append(imginfo)

with open(to_file, "w+") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(imginfos)

