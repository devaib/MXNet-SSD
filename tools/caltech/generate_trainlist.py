import os
import json

to_file = '../../data/caltech-pedestrian-dataset-converter/data/trainlist.txt'
label_path = '../../data/caltech-pedestrian-dataset-converter/data/annotations.json'
train_set = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']
val_set = ['set06', 'set07', 'set08', 'set09', 'set10']

imgnames = []
with open(label_path) as f:
    annos = json.load(f)
    labels = dict()
    people_list = []

    for setname in annos.keys():
        if setname not in train_set:
            continue
        for videoname in annos[setname].keys():
            for imagename in annos[setname][videoname]['frames'].keys():
                imgnames.append("{}_{}_{}".format(setname, videoname, imagename))

with open(to_file, "w+") as f:
    f.writelines("\n".join(imgnames))
