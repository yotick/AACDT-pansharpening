clc; clear all; close all;
%% only for plaidas
addpath(genpath('E:\remote sense image fusion\shared'))

global Qblocks_size   flag_cut_bounds dim_cut thvalues  ratio L sate;

% sate = 'wv3_8';
sate = 'ik';   %% wv3-8  pl  ik
method = 'lu11';

curr_d = pwd;

initialize_sate_RS_MTF();
% time = load(strcat(sate,'_time_RS.mat')).time;
time = zeros(1,100);
if (strcmp(sate, 'wv3_8'))
        path_fused = '.\fused\WorldView-3-8\wv3-8b';   % need to change
        path_gt = 'E:\remote sense image fusion\Source Images\data2017\DIV2K_valid_HR\test_img7\gt256';
        
        files_fused = dir(fullfile(path_fused,'*.mat'));
        files_gt = dir(fullfile(path_gt,'*.mat'));        

%         im_gt = im2double(im_gt);
    elseif (strcmp(sate, 'pl'))
        path_fused = '.\fused\Pleiades\pl_4b';
        path_gt = 'E:\remote sense image fusion\Source Images\data2017\DIV2K_valid_HR\test_img3\gt256';
        
        files_fused = dir(fullfile(path_fused,'*.tif'));
        files_gt = dir(fullfile(path_gt,'*.tif'));        

    elseif (strcmp(sate, 'ik'))   
        path_fused = '.\fused\IKONOS_AIFE\ik_4b';
        path_gt = 'E:\remote sense image fusion\Source Images\data2017\DIV2K_valid_HR\test_img4\gt256';
        
        files_fused = dir(fullfile(path_fused,'*.tif'));
        files_gt = dir(fullfile(path_gt,'*.tif'));      

end
% time = zeros(1,length(files_fused));
num = length(files_fused);
for  i= 1: 100
    %     %     count=count+1
    cd(curr_d);    
    if (strcmp(sate, 'wv3_8'))               
        im_F = load(fullfile(path_fused,files_fused(i).name)).fused;
        im_gt = load(fullfile(path_gt,files_gt(i).name)).gt;
%         im_gt = im2double(im_gt);
    elseif (strcmp(sate, 'pl'))
        im_F = imread(fullfile(path_fused,files_fused(i).name));
        im_gt = imread(fullfile(path_gt,files_gt(i).name));
    elseif (strcmp(sate, 'ik'))           
        im_F = imread(fullfile(path_fused,files_fused(i).name));
        im_gt = imread(fullfile(path_gt,files_gt(i).name));
    end
    
%    imshow(im_F(:,:,1:3));
%    figure, imshow(im_gt(:,:,1:3));
%     im_M=imread(fullfile(savepath_up,file_path_rgb256(i).name));
%     im_pan256=imread(fullfile(folder_pan256,file_path_pan256(i).name));
    
%     im_F = im2double(im_F);
%     im_pan256= im2double(im_pan256);
%     im_M = im2double(im_M);
%     im_gt= im2double(im_gt);
    
    %     [D1,D2,QNR]=getQNR4(im_gt,im_pan256,im_F);
    [Q_avg_Segm, SAM_Segm, ERGAS_Segm, SCC_GT_Segm, Q_Segm] = indexes_evaluation(im_F,im_gt, ...
   ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
    
    [peaksnr] = psnr(im_F,im_gt);
    data{i,1}=files_fused(i).name;
    data{i,2}=roundn(peaksnr,-4);
    data{i,3}=roundn(CC4(im_gt,im_F),-4);
    data{i,4}=roundn(UIQI4(im_gt,im_F),-4);
    data{i,5}=roundn(RASE(im_gt,im_F),-4);
%     data{i,6}=roundn(RMSE(im2double(im_gt),im2double(im_F)),-4);
    data{i,6}=roundn(RMSE(im_gt,im_F),-4);
    data{i,7}=roundn(SAM4(im_gt,im_F),-4);
    data{i,8}=roundn(ERGAS(im_gt,im_F),-4);
    
    data{i,10}=roundn(Q_avg_Segm,-4);
    data{i,11}=roundn(SAM_Segm,-4);
    data{i,12}=roundn(ERGAS_Segm,-4);
    data{i,13}=roundn(SCC_GT_Segm,-4);
    data{i,14}=roundn(Q_Segm,-4);
    data{i,15} = roundn(time(i),-2);
end
j = num +1;
data{j,1}='average';
data{j,2}=roundn(mean(cell2mat(data(1:j-1, 2))),-4);
data{j,3}=roundn(mean(cell2mat(data(1:j-1, 3))),-4);
data{j,4}=roundn(mean(cell2mat(data(1:j-1, 4))),-4);
data{j,5}=roundn(mean(cell2mat(data(1:j-1, 5))),-4);
data{j,6}=roundn(mean(cell2mat(data(1:j-1, 6))),-4);
data{j,7}=roundn(mean(cell2mat(data(1:j-1, 7))),-4);
data{j,8}=roundn(mean(cell2mat(data(1:j-1, 8))),-4);
data{j,10}=roundn(mean(cell2mat(data(1:j-1, 10))),-4);
data{j,11}=roundn(mean(cell2mat(data(1:j-1,11))),-4);
data{j,12}=roundn(mean(cell2mat(data(1:j-1,12))),-4);
data{j,13}=roundn(mean(cell2mat(data(1:j-1,13))),-4);
data{j,14}=roundn(mean(cell2mat(data(1:j-1,14))),-4);
data{j,15}=roundn(mean(cell2mat(data(1:j-1,15))),-2);

T = cell2table(data,'VariableNames',{'FILENAME' 'PSNR' 'CC' 'UIQI' 'RASE' 'RMSE' 'SAM' 'ERGAS'...
    'none' 'Q_UIQI' 'SAM2'  'ERGAS2' 'SCC' 'Q2n' 'time'});
cd(curr_d);
writetable(T,strcat('final_result/quality_',method,'_4C_',sate,'_RS_100_AIFE.xlsx'),'WriteRowNames',true);



