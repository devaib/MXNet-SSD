import json
import os

annoFilePath = '../data/caltech-pedestrian-dataset-converter/data'
annoFile = os.path.join(annoFilePath, 'annotations.json')
anno = json.load(open(annoFile))

outFile = os.path.join(annoFilePath, 'annotations.txt')

with open(outFile, 'w') as outfile:
    json.dump(anno, outfile, indent=2)

