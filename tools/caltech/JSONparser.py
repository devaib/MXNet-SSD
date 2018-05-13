import json
import os

annoFilePath = '../../data/caltech-pedestrian-dataset-converter/data'
annoFile = os.path.join(annoFilePath, 'annotations.json')
#anno = json.load(open(annoFile))
#outFile = os.path.join(annoFilePath, 'annotations.txt')
#with open(outFile, 'w') as outfile:
#    json.dump(anno, outfile, indent=2)

with open(annoFile) as f:
    annos = json.load(f)
    labels = dict()
    people_list = []

    for setname in annos.keys():
        for videoname in annos[setname].keys():
            print setname, videoname
            for imagename in annos[setname][videoname]['frames'].keys():
                image = annos[setname][videoname]['frames'][imagename]
                for detection in image:
                    cls_name = detection['lbl']
                    #if cls_name in labels.keys():
                    #    labels[cls_name] = labels[cls_name] + 1
                    #else:
                    #    labels[cls_name] = 0
                    if cls_name == "people":
                        people_list.append(",".join((setname, videoname, imagename)))

    for people in people_list:
        print people

