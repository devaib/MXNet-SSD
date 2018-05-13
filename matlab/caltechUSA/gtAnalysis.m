data = load('results/UsaTest/gt-Reasonable.mat');
gt = data.gt;
gt_len = size(gt, 2);

gt_arr = [];
for i = 1 : gt_len
    if isempty( gt{1, i} ) == 0
        gt_arr = [ gt_arr; gt{1,i}];
    end
end

gt_ignore = [];
gt_valid = [];
for i = 1 : size(gt_arr, 1)
    if gt_arr(i, 5) == 1
        gt_ignore = [ gt_ignore; gt_arr(i, :) ];
    else
        gt_valid = [ gt_valid; gt_arr(i, :) ];
    end
end

figure(1);
edges = [0:10:500];
subplot(2,1,1); histogram(gt_valid(:,4), edges); title('valid');
subplot(2,1,2); histogram(gt_ignore(:,4), edges); title('ignored');