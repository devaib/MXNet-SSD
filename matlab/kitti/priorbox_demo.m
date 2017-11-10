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
% img = imresize(img, [300, 300]);

img_height = size(img, 1);
img_width  = size(img, 2);
occ_col    = {'g','y','r','w','g','y'};
for i = 1:size(sizes, 1)
    imshow(img);
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
    text(0,00,sprintf('Scale %d', i),'color','g','HorizontalAlignment','left','VerticalAlignment','top','FontSize',14,'FontWeight','bold','BackgroundColor','black');

    waitforbuttonpress; 
    key = get(gcf,'CurrentCharacter');
    switch lower(key)                         
        case 'q',  break;                                 % quit
    end
    
    close; 
end

close all;

