clc; clear; close all;

imagenum = 148;
anchors_dir = './anchors_customized_outputs/';
filename = sprintf('%06d.txt',imagenum);
anchor_file = strcat(anchors_dir, filename);
f = fopen(anchor_file);             
anchors = textscan(f,'%f %f %f %f %f %f %f', 'delimiter', ',');
fclose(f);
index = anchors{1,1};
cls_id = anchors{1,2};
score = anchors{1,3};
ax = anchors{1,4};
ay = anchors{1,5};
aw = anchors{1,6};
ah = anchors{1,7};

[~,ord]=sort(score,'descend');
score(score==-1) = [];
score_sorted = score(ord);
x = ax(ord);