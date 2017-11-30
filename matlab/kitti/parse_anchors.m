clc; clear; close all;
disp('===== Parsing <anchors>.txt and generate statistical results =====');

threshold = 0.5;
num_anchors = 19852;
anchors_dir = './anchors/';
activated_anchors = zeros(num_anchors, 1);
files = dir(strcat(anchors_dir, '*.txt'));
imagenum = 32;  % number of the imagce
feature_map_height = [22 22 11 6 3 2];
feature_map_width = [75 75 38 19 10 5];
anchors_per_location = [4 6 6 6 4 4];

mode = 0;   % 0 - parse anchors, 1 - demo

% demo mode, show valid anchors, only parse one file
if mode == 1
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

    % get valid anchors
    valid_index = index(score > threshold);     % find anchors with score > threshold
    valid_index = valid_index + 1;              % change from 0-indexed to 1-indexed (range from 1 to #anchors)
    activated_anchors(valid_index) = activated_anchors(valid_index) + 1;
    valid_x = ax(valid_index);
    valid_y = ay(valid_index);
    valid_w = aw(valid_index);
    valid_h = ah(valid_index);
    valid_score = score(valid_index);
    
    % load image
    root_dir = '../../data/kitti/data_object_image_2';
    data_set = 'training';
    cam = 2;
    image_dir = fullfile(root_dir,[data_set '/image_' num2str(cam)]);
    img = imread(sprintf('%s/%06d.png',image_dir,imagenum));
    height = size(img, 1);
    width = size(img, 2);
    center_x = round(valid_x .* width);
    center_y = round(valid_y .* height);
    ww = round(valid_w .* width ./ 2);
    hh = round(valid_h .* height ./ 2);
    
    imshow(img);hold on;
    for i = 1:size(valid_index,1)
        pos = [center_x-ww, center_y-hh, 2*ww, 2*hh];
        rect = rectangle('Position', pos(i,:), 'EdgeColor', 'g', 'LineWidth', 2);
        label_text = sprintf('%.2f', valid_score(i));
        txt = text(center_x(i),center_y(i)-hh(i)-6,label_text,'color','g',...
            'BackgroundColor','k','HorizontalAlignment','center',...
            'VerticalAlignment','bottom','FontWeight','bold',...
            'FontSize',8);
        pause(0.2);
        delete(rect); delete(txt);
    end
    
    return
end

% get frequences of activated anchors
if exist('activated_anchors.mat', 'file') == 2
    load('activated_anchors.mat');
else
    counter = 0;
    for file = files'
        if mod(counter, 50) == 0
            sprintf('Processing %d files...', counter+1)
        end
        anchor_file = strcat(anchors_dir, file.name);
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

        valid_index = index(score > threshold);     % find anchors with score > threshold
        valid_index = valid_index + 1;              % change from 0-indexed to 1-indexed (range from 1 to #anchors)
        activated_anchors(valid_index) = activated_anchors(valid_index) + 1;
        counter = counter + 1;
    end
    save('activated_anchors.mat', 'activated_anchors');
end

% generate heatmap
i = 1;
freq_map = cell(1,6);
for scale_ind = 1:size(feature_map_height, 2)
    feature_h = feature_map_height(scale_ind);
    feature_w = feature_map_width(scale_ind);
    freq_map{1,scale_ind} = zeros(feature_h, feature_w);
    anchs_per_loc = anchors_per_location(scale_ind);
    for r = 1:feature_h
        for c = 1:feature_w
            for a = 1:anchs_per_loc
                freq_map{1,scale_ind}(r,c) = freq_map{1,scale_ind}(r,c) + activated_anchors(i);
                i = i + 1;
            end
        end
    end
end

for scale_ind = 1:size(feature_map_height, 2)
    figure;
    colormap('jet');
    imagesc(freq_map{1,scale_ind});
    daspect([1 1 1]);
    colorbar;
    title(sprintf('feature layer %d', scale_ind));
end

