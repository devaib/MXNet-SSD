function frequency_object_w_different_height()
    mode = 1; % 0 - show all, 1/2/3/4- show pairwise comparison between layer 1/2/3/4 and gts
    suffix = '_customized';
    % load gts of validation set
    layer_name = {'first','second','third','fourth'};
    gts_file = '../../data/kitti/results/imginfos_valset.txt';  % load ground truth from validation set
    dts_file1 = strcat('../../data/kitti/results/dts_',layer_name{1},'_layer.txt');
    dts_file2 = strcat('../../data/kitti/results/dts_',layer_name{2},'_layer.txt');
    dts_file3 = strcat('../../data/kitti/results/dts_',layer_name{3},'_layer.txt');
    dts_file4 = strcat('../../data/kitti/results/dts_',layer_name{4},'_layer.txt');
        
    dts_file5 = strcat('../../data/kitti/results/dts_',layer_name{1},'_layer', suffix, '.txt');
    dts_file6 = strcat('../../data/kitti/results/dts_',layer_name{2},'_layer', suffix, '.txt');
    dts_file7 = strcat('../../data/kitti/results/dts_',layer_name{3},'_layer', suffix, '.txt');
    dts_file8 = strcat('../../data/kitti/results/dts_',layer_name{4},'_layer', suffix, '.txt');
    dts_small_file = strcat('../../data/kitti/results/dts_one_layer_customized_small_objects.txt');
    
    f = fopen(gts_file);
    gt_infos = textscan(f,'%s %d %d %d %d %f %d','delimiter',',');
    fclose(f);
    f = fopen(dts_file1);dt_infos1 = textscan(f,'%s %d %d %d %d %f','delimiter',',');fclose(f);
    f = fopen(dts_file2);dt_infos2 = textscan(f,'%s %d %d %d %d %f','delimiter',',');fclose(f);
    f = fopen(dts_file3);dt_infos3 = textscan(f,'%s %d %d %d %d %f','delimiter',',');fclose(f);
    f = fopen(dts_file4);dt_infos4 = textscan(f,'%s %d %d %d %d %f','delimiter',',');fclose(f);
    f = fopen(dts_file5);dt_infos5 = textscan(f,'%s %d %d %d %d %f','delimiter',',');fclose(f);
    f = fopen(dts_file6);dt_infos6 = textscan(f,'%s %d %d %d %d %f','delimiter',',');fclose(f);
    f = fopen(dts_file7);dt_infos7 = textscan(f,'%s %d %d %d %d %f','delimiter',',');fclose(f);
    f = fopen(dts_file8);dt_infos8 = textscan(f,'%s %d %d %d %d %f','delimiter',',');fclose(f);
    f = fopen(dts_small_file);dt_small_info = textscan(f,'%s %d %d %d %d %f','delimiter',',');fclose(f);
    
    gt_heights = gt_infos{1,5};
    dt_heights1 = dt_infos1{1,5};
    dt_heights2 = dt_infos2{1,5};
    dt_heights3 = dt_infos3{1,5};
    dt_heights4 = dt_infos4{1,5};
    dt_heights5 = dt_infos5{1,5};
    dt_heights6 = dt_infos6{1,5};
    dt_heights7 = dt_infos7{1,5};
    dt_heights8 = dt_infos8{1,5};
    dt_small_height = dt_small_info{1,5};
    [y_gt, b_gt] = hist(gt_heights, 0:10:400);
    [y_dt1, b_dt1] = hist(dt_heights1, 0:10:400);
    [y_dt2, b_dt2] = hist(dt_heights2, 0:10:400);
    [y_dt3, b_dt3] = hist(dt_heights3, 0:10:400);
    [y_dt4, b_dt4] = hist(dt_heights4, 0:10:400);
    [y_dt5, b_dt5] = hist(dt_heights5, 0:10:400);
    [y_dt6, b_dt6] = hist(dt_heights6, 0:10:400);
    [y_dt7, b_dt7] = hist(dt_heights7, 0:10:400);
    [y_dt8, b_dt8] = hist(dt_heights8, 0:10:400);
    [y_dt_small, b_dt_small] = hist(dt_small_height, 0:10:400);
    y_dts = [y_dt1; y_dt2; y_dt3; y_dt4];
    y_dts_customized = [y_dt5; y_dt6; y_dt7; y_dt8];
    assert(isequal(b_gt,b_dt1,b_dt2,b_dt3,b_dt4,b_dt5,b_dt6,b_dt7,b_dt8), 'bin dimension not match');
    bin = b_gt;
%     if mode == 0
%         b = bar(bin', [y_dt1; y_dt2; y_dt3; y_dt4; y_gt]', 'grouped');
%         b(1).FaceColor = [0 0 .8];
%         b(2).FaceColor = [0 .8 0];
%         b(3).FaceColor = [.9 .9 0];
%         b(4).FaceColor = [0 0 0];
%         b(5).FaceColor = [1 0 0];
%         set(gcf, 'Position', [100, 500, 1000, 500])
%         xlim([0 400]);
%         xlabel('object height');
%         ylabel('num of object');
%         legend(b, {'detections from layer1','detections from layer2','detections from layer3',...
%             'detections from layer4','ground truth'});
%         title('dts vs gts on each layer');
%     else
        y_dt = y_dts(mode, :);
        y_dt_customized = y_dts_customized(mode, :);
        % b = bar(bin', [y_dt; y_dt_customized; y_gt]', 'grouped');
        b = bar(bin', [y_dt_small; y_dt_customized; y_gt]', 'grouped');
        b(1).FaceColor = [0 0 1];
        b(2).FaceColor = [0 .9 0];
        b(3).FaceColor = [1 0 0];
        set(gcf, 'Position', [100, 500, 1000, 500])
        xlim([0 400]);
        xlabel('object height');
        ylabel('num of object');
        legend(b, {sprintf('detections from customized layer%d trained on small objects',mode),sprintf('detections from customized layer%d',mode),'ground truth'});
        title(sprintf('dts from layer%d vs gts', mode));
%     end
end