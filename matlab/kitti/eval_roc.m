function eval_roc
clear; close; clc;

val_file = '../../data/kitti/data_object_image_2/training/val.txt';
gts_file = '../../data/kitti/results/gts.txt';
dts_file = '../../data/kitti/results/dts.txt';
% dts_file = '../../data/kitti/results/dts_all_layer_customized.txt';
% List of experiment settings: { name, hr, vr, ar, overlap, filter }
%  name     - experiment name
%  hr       - height range to test
%  vr       - visibility range to test
%  ar       - aspect ratio range to test
%  overlap  - overlap threshold for evaluation
%  filter   - expanded filtering (see 3.3 in PAMI11)
exps = {
  'Reasonable',     [50 inf],  [.65 inf], 0,   .5,  1.25
  'All',            [20 inf],  [.2 inf],  0,   .5,  1.25
  'Scale=large',    [100 inf], [inf inf], 0,   .5,  1.25
  'Scale=near',     [80 inf],  [inf inf], 0,   .5,  1.25
  'Scale=medium',   [30 80],   [inf inf], 0,   .5,  1.25
  'Scale=far',      [20 30],   [inf inf], 0,   .5,  1.25
  'Occ=none',       [50 inf],  [inf inf], 0,   .5,  1.25
  'Occ=partial',    [50 inf],  [.65 1],   0,   .5,  1.25
  'Occ=heavy',      [50 inf],  [.2 .65],  0,   .5,  1.25
  'Ar=all',         [50 inf],  [inf inf], 0,   .5,  1.25
  'Ar=typical',     [50 inf],  [inf inf],  .1, .5,  1.25
  'Ar=atypical',    [50 inf],  [inf inf], -.1, .5,  1.25
  'Overlap=25',     [50 inf],  [.65 inf], 0,   .25, 1.25
  'Overlap=50',     [50 inf],  [.65 inf], 0,   .50, 1.25
  'Overlap=75',     [50 inf],  [.65 inf], 0,   .75, 1.25
  'Expand=100',     [50 inf],  [.65 inf], 0,   .5,  1.00
  'Expand=125',     [50 inf],  [.65 inf], 0,   .5,  1.25
  'Expand=150',     [50 inf],  [.65 inf], 0,   .5,  1.50 };
exps=cell2struct(exps',{'name','hr','vr','ar','overlap','filter'});

nn=1000; clrs=zeros(nn,3);
for ii=1:nn, clrs(ii,:)=max(.3,mod([78 121 42]*(ii+1),255)/255); end
algs = {
  'SSD-ResNet101',               0, clrs(1,:),   '-'};
algs=cell2struct(algs',{'name','resize','color','style'});

% remaining parameters and constants
aspectRatio = .41;        % default aspect ratio for all bbs
bnds = [5 5 635 475];     % discard bbs outside this pixel range
plotRoc = 0;              % if true plot ROC else PR curves
plotAlg = 0;              % if true one plot per alg else one plot per exp
plotNum = 15;             % only show best plotNum curves (and VJ and HOG)
samples = 10.^(-2:.25:0); % samples for computing area under the curve
%samples = 0:0.1:1;
lims = [2e-4 50 .035 1];  % axis limits for ROC plots
bbsShow = 0;              % if true displays sample bbs for each alg/exp
bbsType = 'fp';           % type of bbs to display (fp/tp/fn/dt)
algo_name = 'SSD-ResNet101';

% load val list, gts and dts files
f = fopen(val_file);             
val_scan = textscan(f,'%s','delimiter','\n');
fclose(f);
f = fopen(gts_file);             
gts_scan = textscan(f,'%s %d %d %d %d', 'delimiter', ',');
fclose(f);
f = fopen(dts_file);             
dts_scan = textscan(f,'%s %d %d %d %d %f','delimiter',',');
fclose(f);

% construct gts, dts
val_list = val_scan{1,1};
gts_index = gts_scan{1,1}; gts_x = gts_scan{1,2}; gts_y = gts_scan{1,3};
gts_w = gts_scan{1,4}; gts_h = gts_scan{1,5};
dts_index = dts_scan{1,1}; dts_x = dts_scan{1,2}; dts_y = dts_scan{1,3};
dts_w = dts_scan{1,4}; dts_h = dts_scan{1,5}; dts_score = dts_scan{1,6};
num_val = size(val_list, 1);
gts = cell(1,num_val);
dts = cell(1,num_val);
for i_val = 1:num_val
    index = val_list(i_val, 1);
    
    % find index match in gts
    ind_gts = find(strcmp(gts_index, index));
    gts{1,i_val}(:,1) = double(gts_x(ind_gts));
    gts{1,i_val}(:,2) = double(gts_y(ind_gts));
    gts{1,i_val}(:,3) = double(gts_w(ind_gts));
    gts{1,i_val}(:,4) = double(gts_h(ind_gts));
    gts{1,i_val}(:,5) = double(zeros(size(ind_gts)));
    
    % find index match in dts
    ind_dts = find(strcmp(dts_index, index));
    dts{1,i_val}(:,1) = double(dts_x(ind_dts));
    dts{1,i_val}(:,2) = double(dts_y(ind_dts));
    dts{1,i_val}(:,3) = double(dts_w(ind_dts));
    dts{1,i_val}(:,4) = double(dts_h(ind_dts));
    dts{1,i_val}(:,5) = double(dts_score(ind_dts));
end

dataName='KITTI';
plotName=[fileparts(mfilename('fullpath')) '/results/' dataName];
if(~exist(plotName,'dir')), mkdir(plotName); end
gts = {gts}; dts = {dts};
gName = [plotName '/gt' '.mat'];
aName = [plotName '/dt-' algo_name '.mat'];
%save(aName,'dts','-v6');
%save(gName,'gts','-v6');
algo = algo_name;
res = evalAlgs( plotName, algo, exps, gts, dts );

% plot curves and bbs
plotExps( res, plotRoc, plotAlg, plotNum, plotName, ...
samples, lims, reshape([algs.color]',3,[])', {algs.style} );
plotBbs( res, plotName, bbsShow, bbsType );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function res = evalAlgs( plotName, algo, exps, gts, dts )
% Evaluate every algorithm on each experiment
%
% OUTPUTS
%  res    - nGt x nDt cell of all evaluations, each with fields
%   .stra   - string identifying algorithm
%   .stre   - string identifying experiment
%   .gtr    - [n x 1] gt result bbs for each frame [x y w h match]
%   .dtr    - [n x 1] dt result bbs for each frame [x y w h score match]
fprintf('Evaluating: %s\n',plotName); nGt=length(gts); nDt=length(dts);
res=repmat(struct('stra',[],'stre',[],'gtr',[],'dtr',[]),nGt,nDt);
nGt = 1; nDt = 1;
for g=1:nGt
  for d=1:nDt
    gt=gts{g}; dt=dts{d}; n=length(gt); assert(length(dt)==n);
    %stra=algs(d).name; stre=exps(g).name;
    stra=algo; stre='';
    %fName = [plotName '/ev-' stra '.mat'];
    %if(exist(fName,'file')), R=load(fName); res(g,d)=R.R; continue; end
    fprintf('\tExp %i/%i, Alg %i/%i: %s\n',g,nGt,d,nDt,stra);
    hr = exps(g).hr.*[1/exps(g).filter exps(g).filter];
    for ff=1:n, bb=dt{ff}; dt{ff}=bb(bb(:,4)>=hr(1) & bb(:,4)<hr(2),:); end
    [gtr,dtr] = bbGt('evalRes',gt,dt,exps(g).overlap);
    R=struct('stra',stra,'stre',stre,'gtr',{gtr},'dtr',{dtr});
    res(g,d)=R; %save(fName,'R');
  end
end
end

function plotExps( res, plotRoc, plotAlg, plotNum, plotName, ...
  samples, lims, colors, styles )
% Plot all ROC or PR curves.
%
% INPUTS
%  res      - output of evalAlgs
%  plotRoc  - if true plot ROC else PR curves
%  plotAlg  - if true one plot per alg else one plot per exp
%  plotNum  - only show best plotNum curves (and VJ and HOG)
%  plotName - filename for saving plots
%  samples  - samples for computing area under the curve
%  lims     - axis limits for ROC plots
%  colors   - algorithm plot colors
%  styles   - algorithm plot linestyles

% Compute (xs,ys) and score (area under the curve) for every exp/alg
[nGt,nDt]=size(res); xs=cell(nGt,nDt); ys=xs; scores=zeros(nGt,nDt);
for g=1:nGt
  for d=1:nDt
    % score : for ROC curves is the *detection* rate at reference *FPPI*
    [xs{g,d},ys{g,d},~,score] = ...
      bbGt('compRoc',res(g,d).gtr,res(g,d).dtr,plotRoc,samples);
    if(plotRoc), ys{g,d}=1-ys{g,d}; score=1-score; end
    if(plotRoc), score=exp(mean(log(score))); else score=mean(score); end
    scores(g,d)=score;
  end
end

% Generate plots
if( plotRoc ), fName=[plotName 'Roc']; else fName=[plotName 'Pr']; end
stra={res(1,:).stra}; stre={res(:,1).stre}; scores1=round(scores*100);
if( plotAlg ), nPlots=nDt; else nPlots=nGt; end; plotNum=min(plotNum,nDt);
for p=1:nPlots
  % prepare xs1,ys1,lgd1,colors1,styles1,fName1 according to plot type
  if( plotAlg )
    xs1=xs(:,p); ys1=ys(:,p); fName1=[fName stra{p}]; lgd1=stre;
    for g=1:nGt, lgd1{g}=sprintf('%2i%% %s',scores1(g,p),stre{g}); end
    colors1=uniqueColors(1,max(10,nGt)); styles1=repmat({'-','--'},1,nGt);
  else
    xs1=xs(p,:); ys1=ys(p,:); fName1=[fName stre{p}]; lgd1=stra;
    for d=1:nDt, lgd1{d}=sprintf('%2i%% %s',scores1(p,d),stra{d}); end
    kp=[find(strcmp(stra,'VJ')) find(strcmp(stra,'HOG')) 1 1];
    [~,ord]=sort(scores(p,:)); kp=ord==kp(1)|ord==kp(2);
    j=find(cumsum(~kp)>=plotNum-2); kp(1:j(1))=1; ord=fliplr(ord(kp));
    xs1=xs1(ord); ys1=ys1(ord); lgd1=lgd1(ord); colors1=colors(ord,:);
    styles1=styles(ord); f=fopen([fName1 '.txt'],'w');
    for d=1:nDt, fprintf(f,'%s %f\n',stra{d},scores(p,d)); end; fclose(f);
  end
  % plot curves and finalize display
  figure(1); clf; grid on; hold on; n=length(xs1); h=zeros(1,n);
  for i=1:n, h(i)=plot(xs1{i},ys1{i},'Color',colors1(i,:),...
      'LineStyle',styles1{i},'LineWidth',2); end
  if( plotRoc )
    yt=[.05 .1:.1:.5 .64 .8]; ytStr=int2str2(yt*100,2);
    for i=1:length(yt), ytStr{i}=['.' ytStr{i}]; end
    set(gca,'XScale','log','YScale','log',...
      'YTick',[yt 1],'YTickLabel',[ytStr '1'],...
      'XMinorGrid','off','XMinorTic','off',...
      'YMinorGrid','off','YMinorTic','off');
    xlabel('false positives per image','FontSize',14);
    ylabel('miss rate','FontSize',14); axis(lims);
  else
    x=1; for i=1:n, x=max(x,max(xs1{i})); end, x=min(x-mod(x,.1),1.0);
    y=.8; for i=1:n, y=min(y,min(ys1{i})); end, y=max(y-mod(y,.1),.01);
    xlim([0, x]); ylim([y, 1]); set(gca,'xtick',0:.1:1);
    xlabel('Recall','FontSize',14); ylabel('Precision','FontSize',14);
  end
  if(~isempty(lgd1)), legend(h,lgd1,'Location','sw','FontSize',10); end
  % save figure to disk (uncomment pdfcrop commands to automatically crop)
  [o,~]=system('pdfcrop'); if(o==127), setenv('PATH',...
      [getenv('PATH') ':/Library/TeX/texbin/:/usr/local/bin/']); end
  % savefig(fName1,1,'pdf','-r300','-fonts'); close(1); f1=[fName1 '.pdf'];
  % system(['pdfcrop -margins ''-30 -20 -50 -10 '' ' f1 ' ' f1]);
  savefig(fName1,1,'jpeg','-r300','-fonts'); f1=[fName1 '.jpg'];
end

end

function plotBbs( res, plotName, pPage, type )
% This function plots sample fp/tp/fn bbs for given algs/exps
if(pPage==0), return; end; [nGt,nDt]=size(res);
% construct set/vid/frame index for each image
[~,setIds,vidIds,skip]=dbInfo;
k=length(res(1).gtr); is=zeros(k,3); k=0;
for s=1:length(setIds)
  for v=1:length(vidIds{s})
    A=loadVbb(s,v); s1=setIds(s); v1=vidIds{s}(v);
    for ff=skip-1:skip:A.nFrame-1, k=k+1; is(k,:)=[s1 v1 ff]; end
  end
end
for g=1:nGt
  for d=1:nDt
    % augment each bb with set/video/frame index and flatten
    dtr=res(g,d).dtr; gtr=res(g,d).gtr;
    for i=1:k
      dtr{i}(:,7)=is(i,1); dtr{i}(:,8)=is(i,2); dtr{i}(:,9)=is(i,3);
      gtr{i}(:,6)=is(i,1); gtr{i}(:,7)=is(i,2); gtr{i}(:,8)=is(i,3);
      dtr{i}=dtr{i}'; gtr{i}=gtr{i}';
    end
    dtr=[dtr{:}]'; dtr=dtr(dtr(:,6)~=-1,:);
    gtr=[gtr{:}]'; gtr=gtr(gtr(:,5)~=-1,:);
    % get bb, ind, bbo, and indo according to type
    if( strcmp(type,'fn') )
      keep=gtr(:,5)==0; ord=randperm(sum(keep));
      bbCol='r'; bboCol='y'; bbLst='-'; bboLst='--';
      bb=gtr(:,1:4); ind=gtr(:,6:8); bbo=dtr(:,1:6); indo=dtr(:,7:9);
    else
      switch type
        case 'dt', bbCol='y'; keep=dtr(:,6)>=0;
        case 'fp', bbCol='r'; keep=dtr(:,6)==0;
        case 'tp', bbCol='y'; keep=dtr(:,6)==1;
      end
      [~,ord]=sort(dtr(keep,5),'descend');
      bboCol='g'; bbLst='--'; bboLst='-';
      bb=dtr(:,1:6); ind=dtr(:,7:9); bbo=gtr(:,1:4); indo=gtr(:,6:8);
    end
    % prepare and display
    n=sum(keep); bbo1=cell(1,n); O=ones(1,size(indo,1));
    ind=ind(keep,:); bb=bb(keep,:); ind=ind(ord,:); bb=bb(ord,:);
    for f=1:n, bbo1{f}=bbo(all(indo==ind(O*f,:),2),:); end
    f=[plotName res(g,d).stre res(g,d).stra '-' type];
    plotBbSheet( bb, ind, bbo1,'fName',f,'pPage',pPage,'bbCol',bbCol,...
      'bbLst',bbLst,'bboCol',bboCol,'bboLst',bboLst );
  end
end
end

function plotBbSheet( bb, ind, bbo, varargin )
% Draw sheet of bbs.
%
% USAGE
%  plotBbSheet( R, varargin )
%
% INPUTS
%  bb       - [nx4] bbs to display
%  ind      - [nx3] the set/video/image number for each bb
%  bbo      - {nx1} cell of other bbs for each image (optional)
%  varargin - prm struct or name/value list w following fields:
%   .fName    - ['REQ'] base file to save to
%   .pPage    - [1] num pages
%   .mRows    - [5] num rows / page
%   .nCols    - [9] num cols / page
%   .scale    - [2] size of image region to crop relative to bb
%   .siz0     - [100 50] target size of each bb
%   .pad      - [4] amount of space between cells
%   .bbCol    - ['g'] bb color
%   .bbLst    - ['-'] bb LineStyle
%   .bboCol   - ['r'] bbo color
%   .bboLst   - ['--'] bbo LineStyle
dfs={'fName','REQ', 'pPage',1, 'mRows',5, 'nCols',9, 'scale',1.5, ...
  'siz0',[100 50], 'pad',8, 'bbCol','g', 'bbLst','-', ...
  'bboCol','r', 'bboLst','--' };
[fName,pPage,mRows,nCols,scale,siz0,pad,bbCol,bbLst, ...
  bboCol,bboLst] = getPrmDflt(varargin,dfs);
n=size(ind,1); indAll=ind; bbAll=bb; bboAll=bbo;
for page=1:min(pPage,ceil(n/mRows/nCols))
  Is = zeros(siz0(1)*scale,siz0(2)*scale,3,mRows*nCols,'uint8');
  bbN=[]; bboN=[]; labels=repmat({''},1,mRows*nCols);
  for f=1:mRows*nCols
    % get fp bb (bb), double size (bb2), and other bbs (bbo)
    f0=f+(page-1)*mRows*nCols; if(f0>n), break, end
    [col,row]=ind2sub([nCols mRows],f);
    ind=indAll(f0,:); bb=bbAll(f0,:); bbo=bboAll{f0};
    hr=siz0(1)/bb(4); wr=siz0(2)/bb(3); mr=min(hr,wr);
    bb2 = round(bbApply('resize',bb,scale*hr/mr,scale*wr/mr));
    bbo=bbApply('intersect',bbo,bb2); bbo=bbo(bbApply('area',bbo)>0,:);
    labels{f}=sprintf('%i/%i/%i',ind(1),ind(2),ind(3));
    % normalize bb and bbo for siz0*scale region, then shift
    bb=bbApply('shift',bb,bb2(1),bb2(2)); bb(:,1:4)=bb(:,1:4)*mr;
    bbo=bbApply('shift',bbo,bb2(1),bb2(2)); bbo(:,1:4)=bbo(:,1:4)*mr;
    xdel=-pad*scale-(siz0(2)+pad*2)*scale*(col-1);
    ydel=-pad*scale-(siz0(1)+pad*2)*scale*(row-1);
    bb=bbApply('shift',bb,xdel,ydel); bbN=[bbN; bb]; %#ok<AGROW>
    bbo=bbApply('shift',bbo,xdel,ydel); bboN=[bboN; bbo]; %#ok<AGROW>
    % load and crop image region
    sr=seqIo(sprintf('%s/videos/set%02i/V%03i',dbInfo,ind(1),ind(2)),'r');
    sr.seek(ind(3)); I=sr.getframe(); sr.close();
    I=bbApply('crop',I,bb2,'replicate');
    I=uint8(imResample(double(I{1}),siz0*scale));
    Is(:,:,:,f)=I;
  end
  % now plot all and save
  prm=struct('hasChn',1,'padAmt',pad*2*scale,'padEl',0,'mm',mRows,...
    'showLines',0,'labels',{labels});
  h=figureResized(.9,1); clf; montage2(Is,prm); hold on;
  bbApply('draw',bbN,bbCol,2,bbLst); bbApply('draw',bboN,bboCol,2,bboLst);
  savefig([fName int2str2(page-1,2)],h,'png','-r200','-fonts'); close(h);
  if(0), save([fName int2str2(page-1,2) '.mat'],'Is'); end
end
end

end

