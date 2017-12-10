clear; close all; clc;

threshold = 0.5;
multiple_match = 0;
filepath = '../../data/kitti/results/';
gts_file = strcat(filepath, 'gts.txt');
dts_file = strcat(filepath, 'dts_all_layer_customized.txt');

% load file
f1 = fopen(gts_file); gts = textscan(f1,'%s %f %f %f %f','delimiter',','); fclose(f1);
f2 = fopen(dts_file); dts = textscan(f2, '%s %f %f %f %f %f', 'delimiter', ','); fclose(f2);
num_gts = size(gts{1,1}, 1);
num_dts = size(dts{1,1}, 1);
gts_index = gts{1,1};
dts_index = dts{1,1};
unique_index = unique(gts_index, 'stable');         % tuple of all image index in gts
num_image = size(unique_index, 1);                  % number of images in gts

gtss = cell(1, num_image);
dtss = cell(1, num_image);
jj = 1; kk = 1;
for i = 1 : num_image
    imgname = unique_index{i,1};                    % string - image name
    find_match = 0;                                 % flag if a match was found
    all_matched = 0;                                % flag if all matches were found
    for j = jj : num_gts                             % find gts of that image
        if all_matched == 1
            break;
        end
        if isequal(gts_index{j, 1}, imgname)        % a match is found
            if find_match == 0
                find_match = 1;
            end
            gts_obj = [gts{1,2}(j) gts{1,3}(j) gts{1,4}(j) gts{1,5}(j) 0.0];
            gtss{1,i} = [gtss{1,i}; gts_obj];
            jj = jj + 1;
        elseif find_match == 1                      % all matches are found
            all_matched = 1;
        end
    end
    find_match = 0;
    all_matched = 0;
    for k = kk : num_dts                           % find dts of that image
        if all_matched == 1
            break;
        end
        if isequal(dts_index{k,1}, imgname)
            if find_match == 0
                find_match = 1;
            end
            dts_obj = [dts{1,2}(k) dts{1,3}(k) dts{1,4}(k) dts{1,5}(k) dts{1,6}(k)];
            dtss{1,i} = [dtss{1,i}; dts_obj];
            kk = kk + 1;
        elseif find_match == 1
            all_matched = 1;
        end
    end
end

[gtr,dtr] = bbGt('evalRes',gtss,dtss,threshold,multiple_match);

root_dir = '../../data/kitti/data_object_image_2';
data_set = 'training';
cam = 2;
image_dir = fullfile(root_dir,[data_set '/image_' num2str(cam)]);
colors = ['r', 'g', 'b', 'y'];

gts_missed = [];        % all missed gts
for i = 1:num_image
    gt_missed = gtr{1,i}(gtr{1,i}(:,5) == 0, :);
    imgname = unique_index{i};
    gts_missed = [gts_missed; gt_missed];
    
%     if size(gt_missed,1) < 1
%         continue
%     end
%     demo_img = imread(sprintf('%s/%s.png',image_dir,imgname));
%     imshow(demo_img); title('Miss detections demo'); hold on;
%     for j = 1:size(gt_missed,1)
%         pos = gt_missed(j,1:4);
%         rect = rectangle('Position', pos, 'EdgeColor', colors(2), 'LineWidth', 2);
%         waitforbuttonpress; %pause(0.2);
%         delete(rect);
%     end
%     close;
end

size_thres = 40;
gts_missed_small = gts_missed(gts_missed(:,4) <= size_thres, :);
% show missed gts with height smaller than 40 
demo_img = imread(sprintf('%s/%s.png',image_dir,imgname));
imshow(demo_img); title('Miss detections demo'); hold on;
for i = 1:size(gts_missed_small,1)
    pos = gts_missed_small(i,1:4);
    rect = rectangle('Position', pos, 'EdgeColor', colors(2), 'LineWidth', 2);
    %waitforbuttonpress; %pause(0.2);
end

% ymins = gts_missed_small(:,2);
% ymaxs = gts_missed_small(:,2) + gts_missed_small(:,4);
% xmins = gts_missed_small(:,1);
% xmaxs = gts_missed_small(:,1) + gts_missed_small(:,3);
% histogram(ymins);


