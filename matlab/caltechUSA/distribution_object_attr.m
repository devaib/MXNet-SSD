% inputs
% attribute : which attribute of image to be explored, 'width', 'height' or 'ratio'
% nBin : number of bins in histogram
function distribution_object_attr(attribute, nBin)
imginfo_file = '../../data/caltech-pedestrian-dataset-converter/results/imginfos.txt';
imginfo_train_file = '../../data/caltech-pedestrian-dataset-converter/results/imginfos_train.txt';
imginfo_val_file = '../../data/caltech-pedestrian-dataset-converter/results/imginfos_val.txt';

% load file
f = fopen(imginfo_file);
imginfos = textscan(f,'%s %s %s %f %f %f %f %f','delimiter',',');
fclose(f);
f = fopen(imginfo_train_file);
imginfos_train = textscan(f,'%s %s %s %f %f %f %f %f','delimiter',',');
fclose(f);
f = fopen(imginfo_val_file);
imginfos_val = textscan(f,'%s %s %s %f %f %f %f %f','delimiter',',');
fclose(f);

imginfo = {1,3};
imginfo{1,1} = imginfos; imginfo{1,2} = imginfos_train; imginfo{1,3} = imginfos_val;
titles = {1,3};
titles{1,1} = 'all'; titles{1,2} = 'train'; titles{1,3} = 'val';

figure(1);
for i = 1 : 3
    if strcmp(attribute, 'width')
        attr = imginfo{1,i}{1,6};
    elseif strcmp(attribute, 'height')
        attr = imginfo{1,i}{1,7};
    elseif strcmp(attribute, 'ratio')
        attr = imginfo{1,i}{1,8};
    end
    %figure
    subplot(3,1,i); 
    %edges = [0:0.01:1];
    %histogram(attr, edges);
    histogram(attr, nBin)
    title(titles{1,i});
    [N, edges] = histcounts(attr, nBin);
    average = mean(attr)
    median_v = median(attr)
    hold on;
    line([average, average], ylim, 'LineWidth', 2, 'Color', 'r');
    line([median_v, median_v], ylim, 'LineWidth', 2, 'Color', 'b');
end

end