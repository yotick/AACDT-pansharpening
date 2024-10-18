clc; clear all; close all;
addpath(genpath('F:\remote sense image fusion\shared'))

global  thvalues  ratio L  sensor sate;

tag = 'China';

% sate = 'wv3_8';
sate = 'pl';   %% wv3-8  pl  ik
method = 'lu11';
curr_d = pwd;

initialize_sate_FS_100();
time = zeros(1,100);
% time = load(strcat(sate,'_time_Full.mat')).time;

%% setting path and files
if (strcmp(sate, 'wv3_8'))
    
    path_fused = '.\fused\WorldView-3-8\wv3-8b-FS';
    path_ms = 'F:\remote sense image fusion\Source Images\data2017\DIV2K_valid_HR\test_img7\gt256';
    path_pan =  'F:\remote sense image fusion\Source Images\data2017\DIV2K_valid_HR\test_img7\PAN\PAN1024';
    
    files_fused = dir(fullfile(path_fused,'*.mat'));
    files_ms = dir(fullfile(path_ms,'*.mat'));
    files_pan = dir(fullfile(path_pan,'*.tif'));
    
    %         im_gt = im2double(im_gt);
elseif (strcmp(sate, 'pl'))
    path_fused = '.\fused\Pleiades\pl_4b_FS';
    path_ms = 'E:\remote sense image fusion\Source Images\data2017\DIV2K_valid_HR\test_img3\gt256';
    path_pan =  'E:\remote sense image fusion\Source Images\data2017\DIV2K_valid_HR\test_img3\PAN\PAN1024';
    
    files_fused = dir(fullfile(path_fused,'*.tif'));
    files_ms = dir(fullfile(path_ms,'*.tif'));
    files_pan = dir(fullfile(path_pan,'*.tif'));
    
elseif (strcmp(sate, 'ik'))
    path_fused = '.\fused\IKONOS\ik_4b_FS';
    path_ms = 'E:\remote sense image fusion\Source Images\data2017\DIV2K_valid_HR\test_img4\gt256';
    path_pan =  'E:\remote sense image fusion\Source Images\data2017\DIV2K_valid_HR\test_img4\PAN\PAN1024';
    
    files_fused = dir(fullfile(path_fused,'*.tif'));
    files_ms = dir(fullfile(path_ms,'*.tif'));
    files_pan = dir(fullfile(path_pan,'*.tif'));
end

% time = zeros(1,length(files_fused));
num = length(files_fused);
eval_times = 0;

%% read data from files
for  i= 1: length(files_fused)
% for  i= 74: 74
% for  i= 1: 5  
    %     %     count=count+1
    cd(curr_d);
    eval_times = eval_times + 1
    if (strcmp(sate, 'wv3_8'))
        im_F = load(fullfile(path_fused,files_fused(i).name)).fused;
        mul_noR = load(fullfile(path_ms,files_ms(i).name)).gt;
        im_pan = imread(fullfile(path_pan,files_pan(i).name));
        %         im_gt = im2double(im_gt);
    elseif (strcmp(sate, 'pl'))
        im_F = imread(fullfile(path_fused,files_fused(i).name));
        mul_noR = imread(fullfile(path_ms,files_ms(i).name));
        im_pan = imread(fullfile(path_pan,files_pan(i).name));
        
    elseif (strcmp(sate, 'ik'))
        im_F = imread(fullfile(path_fused,files_fused(i).name));
        mul_noR = imread(fullfile(path_ms,files_ms(i).name));
        im_pan = imread(fullfile(path_pan,files_pan(i).name));
    end
    
    ms_up = imresize(mul_noR,4);
    im_F = im2double(im_F);
    im_pan= im2double(im_pan);
    mul_noR= im2double(mul_noR);
    ms_up = im2double(ms_up);
    
%     [D_lambda_Segm,D_S_Segm,QNRI_Segm] = indexes_evaluation_FS2(im_F,mul_noR,im_pan,...
%         L,thvalues,ms_up,sensor,ratio, 1);
    [D_lambda,D_S,QNR_index,SAM_index,sCC] = indexes_evaluation_FS(im_F,mul_noR,im_pan,L,thvalues,ms_up,sensor,tag,ratio);
    
    [D_lambda_Segm2,D_S_Segm2,HQNR] = indexes_evaluation_FS2(im_F,mul_noR,im_pan,...
        L,thvalues,ms_up,sensor,ratio, 0);
    
    data{i,1}=files_fused(i).name;    
    data{i,2}=roundn(D_lambda,-4);
    data{i,3}=roundn(D_S,-4);
    data{i,4}=roundn(QNR_index,-4);    
    
    data{i,6}=roundn(D_lambda_Segm2,-4);
    data{i,7}=roundn(D_S_Segm2,-4);
    data{i,8}=roundn(HQNR,-4);
    data{i,9} = roundn(time(i),-2);
end
%% average 
j = num +1;
data{j,1}='average';
data{j,2}=roundn(mean(cell2mat(data(1:j-1, 2))),-4);
data{j,3}=roundn(mean(cell2mat(data(1:j-1, 3))),-4);
data{j,4}=roundn(mean(cell2mat(data(1:j-1, 4))),-4);
data{j,6}=roundn(mean(cell2mat(data(1:j-1, 6))),-4);
data{j,7}=roundn(mean(cell2mat(data(1:j-1, 7))),-4);
data{j,8}=roundn(mean(cell2mat(data(1:j-1, 8))),-4);
data{j,9}=roundn(mean(cell2mat(data(1:j-1, 9))),-2);

T = cell2table(data,'VariableNames',{'FILENAME' 'D_lambda' 'D_S' 'QNRI' 'none' 'D_lambda2' 'D_S2' 'HQNR' 'time'});
cd(curr_d);
writetable(T,strcat('final_result/quality_',method,'_4C_',sate,'_Full_100.xlsx'),'WriteRowNames',true);



