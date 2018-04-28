% This is test script for logistic regression 

close all;
clear;
clc;

% nba_data is a matrix 
% each row denotes the performance of a certain NBA team, which contains 
% scores, assits, backboard, etc. The last column indicate the team win 
% champions or not
load nbadata;

% Before training on real dataset, we test our algorithm on a simple dataset 
x = [0,0;2,2;2,0;3,0];
y = [0;0;1;1];
c = [1;1;1;1];
x_hom = [c x]; % homogeneous form

%% write your own function for logistic regression as lr_yourname.m 

[ weight, glist ] = logisRegression(x_hom, y);

% input:
%      x_hom:    data matrix with homogeneous form
%       y:      label, a vector
%
% output:
%      weight:  parameters in logistic regression weight = [b, w]
%      glist:   record the norm of gradient in iteration, a vector





%% plot ||\nabla g|| of simple dataset
semilogy(glist);
set(gca,'FontSize',15)
xlabel('iteration','FontSize',15)
ylabel('log ||\nabla g||', 'FontSize',15)


nba_datahom = [ones(size(nba_data,1), 1) nba_data(:,1:end-1)];
cham_label = nba_data(:,end);

%% write your own function for logistic regression as lr_yourname.m

[ weight, glist ] = logisRegression(nba_datahom, cham_label);

% input:
%      x_hom:    data matrix with homogeneous form
%       y:      label, a vector
%
% output:
%      weight:  parameters in logistic regression weight = [b, w]
%      glist:   record the norm of gradient in iteration, a vector




% compute the accuracy of prediction in training set
pred_accuracy = 




