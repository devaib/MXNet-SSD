function visualize_detections()
    suffix1 = '_customized';
    suffix2 = '_customized_central';
    % load gts of validation set
    layer_name = {'first','second','third','fourth','all'};
    gts_file = '../../data/kitti/results/imginfos_valset.txt';  % load ground truth from validation set
    dts_file = strcat('../../data/kitti/results/dts_',layer_name{5},'_layer', suffix1, '.txt');
    dts_central_file = strcat('../../data/kitti/results/dts_',layer_name{5},'_layer', suffix2, '.txt');
    val_file = '/home/binghao/workspace/MXNet-SSD/data/kitti/data_object_image_2/training/val.txt';
    f = fopen(val_file); val_list = textscan(f,'%s','delimiter','\n'); fclose(f);
    f = fopen(gts_file); gt_infos = textscan(f,'%s %f %f %f %f %f %f','delimiter',','); fclose(f);
    f = fopen(dts_file); dt_infos = textscan(f,'%s %f %f %f %f %f','delimiter',','); fclose(f);
    f = fopen(dts_central_file); dt_central_infos = textscan(f,'%s %f %f %f %f %f','delimiter',','); fclose(f);
    cropped_image_dir = '../../data/kitti/data_object_image_2/training/image_2/';
    
    val_lst = val_list{1,1};
    imgindices = dt_infos{1,1};
    xs = dt_infos{1,2};
    ys = dt_infos{1,3};
    ws = dt_infos{1,4};
    hs = dt_infos{1,5};
    scores = dt_infos{1,6};
    imgindices_central = dt_central_infos{1,1};
    xs_central = dt_central_infos{1,2};
    ys_central = dt_central_infos{1,3};
    ws_central = dt_central_infos{1,4};
    hs_central = dt_central_infos{1,5};
    scores_central = dt_central_infos{1,6};
    i = 1; k = 1;
    for ind = 1 : size(val_lst, 1)
        imgindex = val_lst{ind}
        j = i;
        imgpath = strcat(cropped_image_dir, imgindex, '.png');
        img = imread(imgpath); imshow(img); hold on;
        width = size(img, 2); height = size(img, 1);
        while strcmp(imgindices{j}, imgindex) == 1
            j = j+1;
        end
        for ii = i:(j-1)
            x = xs(ii); y = ys(ii); w = ws(ii); h = hs(ii); score = scores(ii);
            pos = [x, y, w, h];
            rect1 = rectangle('Position', pos, 'EdgeColor', 'g', 'LineWidth', 2);
            label_text = sprintf('%.2f', score);
            txt1 = text(x,y+h+12,label_text,'color','g',...
            'BackgroundColor','k','HorizontalAlignment','center',...
            'VerticalAlignment','bottom','FontWeight','bold',...
            'FontSize',6);
        end
        i = j;
        
        m = k;
        while strcmp(imgindices_central{m}, imgindex) == 1
            m = m+1;
        end
        for kk = k:(m-1)
            x = xs_central(kk); y = ys_central(kk); w = ws_central(kk); h = hs_central(kk); score = scores_central(kk);
            ww = w / 2; hh = h / 2; xx = x + ww; yy = y + hh;
            xx = xx / 2 + 0.25 * width; yy = yy / 2 + 0.25 * height; ww = ww / 2; hh = hh / 2;
            x = xx - ww; y = yy - hh; w = ww * 2; h = hh * 2;
            pos = [x, y, w, h];
            rect2 = rectangle('Position', pos, 'EdgeColor', 'r', 'LineWidth', 2);
            label_text = sprintf('%.2f', score);
            txt2 = text(x+w,y-6,label_text,'color','r',...
            'BackgroundColor','k','HorizontalAlignment','center',...
            'VerticalAlignment','bottom','FontWeight','bold',...
            'FontSize',6);
        end
        k = m;
        
        waitforbuttonpress;
        %saveas(fig, strcat(imgindex, '.jpg'));
        delete(rect1); delete(txt1);delete(rect2); delete(txt2); close;
    end

%     % load gts of validation set
%     cropped_image_dir = '../../data/kitti/data_object_image_2/training/image_2/';
%     imgindex = sprintf('%06d', 1350);
%     imgpath = strcat(cropped_image_dir, imgindex, '.png');
%     img = imread(imgpath); imshow(img); hold on;
%     height = size(img,1); width = size(img,2);
%     detections = [.449 .447 .464 .493];
%     detections(:,3) = detections(:,3) - detections(:,1);
%     detections(:,4) = detections(:,4) - detections(:,2);
%     detections(:,[1 3]) = detections(:,[1 3]) * width;
%     detections(:,[2 4]) = detections(:,[2 4]) * height;
%     for i = 1 : size(detections, 1)
%         x = detections(i,1); y = detections(i,2); w = detections(i,3); h = detections(i,4);
%         pos = [x, y, w, h];
%         rect = rectangle('Position', pos, 'EdgeColor', 'g', 'LineWidth', 2);
%     end
%     waitforbuttonpress;

    close all;
end