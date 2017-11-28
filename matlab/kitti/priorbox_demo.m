clc; clear; close all;
disp('===== Priorbox Generation Demo =====');

sizes = [.1,  .141; 
         .2,  .272; 
         .37, .447; 
         .54, .619; 
         .71, .79; 
         .88, .961];
ratios = [1,2,.5,-1,-1; 
    1,2,.5,3,1./3; 
    1,2,.5,3,1./3; 
    1,2,.5,3,1./3; 
    1,2,.5,-1,-1; 
    1,2,.5,-1,-1];

w = -1 .* ones(size(sizes, 1), size(sizes,2)+size(ratios,2)-1);
h = -1 .* ones(size(sizes, 1), size(sizes,2)+size(ratios,2)-1);
for i = 1:6
    size_i = sizes(i, :);
    ratio = ratios(i, :);

    for j = 1 : size(size_i, 2)
        w(i, j) = size_i(j) / 2;
        h(i, j) = size_i(j) / 2;
    end
    for k = 2 : size(ratio, 2)
        if ratio(k) < 0
            continue; 
        end
        w(i, size(size_i,2)+k-1) = size_i(1) * sqrt(ratio(k)) / 2;
        h(i, size(size_i,2)+k-1) = size_i(1) / sqrt(ratio(k)) / 2;
    end
end
root_dir = '/home/binghao/data/kitti/data_object_image_2';
data_set = 'training';
cam = 2;
image_dir = fullfile(root_dir,[data_set '/image_' num2str(cam)]);
img = imread(sprintf('%s/%06d.png',image_dir,8));
%img = imresize(img, [300, 300]);
%feature_map_height = [19 19 10 5 3 2];
%feature_map_width = [19 19 10 5 3 2];

feature_map_height = [22 22 11 6 3 2];
feature_map_width = [75 75 38 19 10 5];
    
img_height = size(img, 1);
img_width  = size(img, 2);
occ_col    = {'g','y','r','w','g','y'};
for i = 1:size(sizes, 1)
    img = imread(sprintf('%s/%06d.png',image_dir,8));
    %img = imresize(img, [300, 300]);
    yy = round(linspace(1,size(img,1),feature_map_height(i)+1));
    xx = round(linspace(1,size(img,2),feature_map_width(i)+1)); 
    step_y = 1 / feature_map_height(i);
    step_x = 1 / feature_map_width(i);
    center_y = zeros(1, int32(feature_map_height(i)));
    center_x = zeros(1, int32(feature_map_width(i)));
    offset_y = 0.5;
    offset_x = 0.5;
    for r = 0 : uint32(feature_map_height(i)-1)
        center_y(r+1) = (double(r) + offset_y) * step_y;
    end
    for c = 0 : uint32(feature_map_width(i)-1)
        center_x(c+1) = (double(c) + offset_x) * step_x;
    end
    center_x = center_x .* img_width;
    center_y = center_y .* img_height;
    imshow(img);hold on;
    for y = 1:size(center_y,2)
        for x = 1:size(center_x,2)
             plot(center_x(x), center_y(y), 'rx');
        end
    end 
   
    text(100, -20, sprintf('feature map size: %d x %d', feature_map_height(i), feature_map_width(i)),'color','g','HorizontalAlignment','left','VerticalAlignment','top','FontSize',10,'FontWeight','bold','BackgroundColor','black')
    for j = 1:size(sizes,2)+size(ratios,2)-1
        if w(i,j) < 0
            continue
        end
        
        center_x = round(img_width  * 0.5 - 1);
        center_y = round(img_height * 0.5 - 1);
        ww       = round(img_width  * w(i,j) - 1);
        hh       = round(img_height * h(i,j) - 1);
        pos = [center_x-ww, center_y-hh, 2*ww, 2*hh];
        rectangle('Position', pos, 'EdgeColor', occ_col{j}, 'LineWidth', 2);
    end
    text(10,-20,sprintf('Scale %d', i),'color','g','HorizontalAlignment','left','VerticalAlignment','top','FontSize',10,'FontWeight','bold','BackgroundColor','black');

    waitforbuttonpress; 
    key = get(gcf,'CurrentCharacter');
    switch lower(key)                         
        case 'q',  break;                                 % quit
    end
    
    close; 
end

close all;

