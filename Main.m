%% Summary of Vibration Statistics for the four Engines
summary(data(:,{'Vib_E1','Vib_E2','Vib_E3', 'Vib_E4'}))
% Max Vibration greater in Engine 1 and 2 compared to 3 and 4.

% Scatter Plot of N1 RPM of Engine1 vs Fuel Flow
scatter(data.N1_F1,data.FF_E1)
xlabel('N1 RPM'), ylabel('Fuel Flow Engine1')
title('Fan RPM vs Fuel Flow Engine1')

%% High EGT of all engines > 575

EGT_all=[data.EGT_E1;data.EGT_E2;data.EGT_E3;data.EGT_E4];
idxe=EGT_all>575;
EGT_all=EGT_all(idxe); %All EGTs higher then 575

%% Different Phases of the Aircraft
%% find unique Phases present in Data
[ph] = unique(data.Phase)

%Cruise is Phase5
idx1=data.Phase==5;
Cruise=data.Phase(idx1);

% idx2=Climb is Phase4
idx2=data.Phase==4;
Climb=data.Phase(idx2);

%Taxi is Phase2
idx3=data.Phase==2;
Taxi=data.Phase(idx3);

%Approach is Phase6
idx4=data.Phase==6;
Approach=data.Phase(idx4);

%Takeoff is Phase3
idx5=data.Phase==3;
Takeoff=data.Phase(idx5);

%% Group Scatter Plot

% Group Scatter at various Phases (Phases is the Categorical Grouping
% Vaiable
figure
gscatter(data.N2_E1, data.FF_E1,data.Phase,'', '.', 10, 'on', 'N2 Engine1', 'Fuel Flow Engine1')
title('N2 vs Fuel Flow at various Flight Phases(Engine1)')

% Group Scatter at various Thrust Modes
figure
gscatter(data.N2_E2, data.FF_E2,data.THRUSTMODE,'', '.', 10, 'on', 'N2_Engine1', 'Fuel Flow Engine1')
title('N2 vs Fuel Flow at various Thrust Modes (Engine2)')

%% Grouped Statistics

% Perform group statistics based on specified grouping variables.

varfun(@mean, data,'InputVariables',{'Vib_E1', 'Vib_E2','Vib_E3'},...
    'GroupingVariables',{'Phase'})

%% Grouped Data Scatter Plot Matrix

figure
gplotmatrix([data.N2_E4, data.PLA_E4], [data.Oil_PressE4, data.FF_E4], ...
    {data.Above, data.THRUSTMODE}, ...
    '', '.', 10, 'on', '', {'N2 Engine4', 'PowerLeverAngle Engine4'}, {'Oil Pressure', 'FuelFlow'})
title('Grouped Scatter Matrix')

%% Box Plot of EGT Distribution at all Phases showing the outliers

figure
boxplot([data.EGT_E1, data.EGT_E2, data.EGT_E3, data.EGT_E4],'notch','on',...
        'labels',{'EGT_Engine1','EGT_Engine2','EGT_Engine3','EGT_Engine4'})
    
%%
% Mean Vibration of Engines at Cruise Phase

%Mean Vibration of Engine
n=(1:4)
a1=mean(data.Vib_E1(idx1)) 
a2=mean(data.Vib_E2(idx1))
a3=mean(data.Vib_E3(idx1))
a4=mean(data.Vib_E4(idx1))
a=[a1,a2,a3,a4]

% Trim Mean 20%
b1=trimmean(data.Vib_E1(idx1),20)
b2=trimmean(data.Vib_E2(idx1),20)
b3=trimmean(data.Vib_E3(idx1),20)
b4=trimmean(data.Vib_E4(idx1),20)
b=[b1,b2,b3,b4]

subplot(1,2,1)
bar(n,a,0.4)
xlabel('Mean Vibration of Engines 1-4')
title('Not Trimmed')
subplot(1,2,2)
bar(n,b,0.4,'r')
xlabel('Trimmed Mean Vibration of Engines 1-4')
title('Trimmed Mean')

%% Standard deviation and Skewness

a1=std(data.Vib_E1(idx1)),b1=skewness(data.EGT_E1(idx1))
a2=std(data.Vib_E2(idx1)),b2=skewness(data.EGT_E2(idx1))
a3=std(data.Vib_E3(idx1)),b3=skewness(data.EGT_E3(idx1))
a4=std(data.Vib_E4(idx1)),b4=skewness(data.EGT_E4(idx1))
figure
subplot(1,2,1)
bar(1:4,[a1,a2,a3,a4],0.4,'g')
xlabel('Standard Deviation of Vibrations')
title('Standard Deviation')
subplot(1,2,2)
bar(1:4,[b1,b2,b3,b4],0.4)
xlabel('Skewness of Vibrations')
title('Skewness')

%%
% EGT Distribution of Engines at Cruise
figure
subplot(2,2,1)
histfit(data.EGT_E1(idx1),100,'kernel'),xlabel('Engine1')
subplot(2,2,2)
histfit(data.EGT_E2(idx1),100,'kernel'),xlabel('Engine2')
subplot(2,2,3)
histfit(data.EGT_E3(idx1),100,'kernel'),xlabel('Engine3')
subplot(2,2,4)
histfit(data.EGT_E4(idx1),100,'kernel'),xlabel('Engine4')

%% 
% Normal Probability Plot

figure
subplot(1,2,1)
hist(data.TotalAir_Temp(idx2),50,'kernel'),xlabel('Total Air Temperature')
title('Histogram of Total Air Temperature at Climb')
subplot(1,2,2)
probplot(data.TotalAir_Temp(idx2)),xlabel('Normal Porbability Plot Air Temperature')

%% Scatter 3D
% Vibration of Engine3 as a function of N2 & EGT 
ffe3=data.FF_E3(idx2); %For color axis
createfigure(data.N2_E3(idx2),data.EGT_E3(idx2),data.Vib_E3(idx2),20,ffe3)
%createfigure function saved

%% ANOVA for Engine Vibration at Approach
v1=data.Vib_E1(idx4);
v2=data.Vib_E2(idx4);
v3=data.Vib_E3(idx4);
v4=data.Vib_E4(idx4);
boxplot([v1,v2,v3,v4],'notch','on',...
    'labels',{'Vibration_Engine1','Vibration_Engine2','Vibration_Engine3','Vibration_Engine4'})
title('Engine Vibrations at Approach Phase')

% N-way analysis of variance
p = anovan(data.Fuel_T4,{data.Phase,data.THRUSTMODE,data.Above,data.WoW},...
'varnames', {'Phases', 'ThrustMode','FreezingPt','WeightWheels'}); 

%% Curve Fitting
N1=data.N1_F4(idx1);
N2=data.N2_E4(idx1);
EGT=data.EGT_E4(idx1);
createFit(N1, N2, EGT) %Function saved

%% Covariance Matrix
% How correlated are each of the data columns?
predic=[data.EGT_E1,data.FF_E1,data.Fuel_T1,data.N1_F1,data.N2_E1,data.Oil_PressE1,data.Oil_TempE1,...
    data.TotalAir_Temp, data.Tot_Pressure,data.Vib_E1, data.Wind_Speed];

y=data.Phase;

%% Data Set Partition for Classification Training and Test Sets

c = cvpartition(y,'holdout');
Xtrain=predic(training(c,1),:);
Xtest=predic(test(c,1),:);
Ytrain=y(training(c,1));
Ytest=y(test(c,1));
%% What var's are important? Feature Selection
% To remove highly correlated variables and improve memory and speed
% opts = statset('display','iter', 'useparallel', 'always');
% tic
%fun = @(Xtrain,Ytrain,Xtest,Ytest)...
%     sum(Ytest~=predict(NaiveBayes.fit(Xtrain,Ytrain,'Distribution','kernel'),Xtest));
%[fs,history] = sequentialfs(fun,predic,y,'cv',c,'options',opts);
%toc;
%% Sequential Feature Selection

%  Selects a subset of features from the data matrix X that best predict the data 
% in y by sequentially selecting features until there is no improvement in prediction.
% Rows of X correspond to observations; columns correspond to variables or features

% Final columns included:  3 4 9 
predic_all=[data.FF_E1,data.Fuel_T1, data.N1_F1, data.N2_E1, data.Oil_PressE1,...
    data.Oil_TempE1, data.Phase, data.PLA_E1, data.Vib_E1, data.THRUSTMODE, data.Wind_Speed,...
    data.TotalAir_Temp, data.Tot_Pressure];
y1=data.EGT_E1;
predic=[data.Fuel_T1,data.N1_F1,data.Tot_Pressure];
app=[predic,y];

%% Perform Exploratory Data Analysis to check assumptions

% See whether the features are normally distribution
% (If the features aren't normally distributed, we shouldn't use
% discriminant analysis)
X=[data.FF_E1,data.Fuel_T1, data.N1_F1, data.N2_E1, data.Oil_PressE1,...
    data.Oil_TempE1, data.Vib_E1, data.TotalAir_Temp, data.Tot_Pressure];
VarNames={'FF_E1','Fuel_T1', 'N1_F1', 'N2_E1', 'Oil_PressE1',...
    'Oil_TempE1', 'Vib_E1', 'TotalAir_Temp', 'Tot_Pressure'}
figure
for i = 1:9
   
   subplot(3,3,i)
   normplot(double(X(:,i))) 
   title(VarNames(i))
   
end
%%
% Covariance Matrix
covmat = corrcoef(double(X));

figure
x = size(X, 2);
imagesc(covmat);
set(gca,'XTick',1:x);
set(gca,'YTick',1:x);
set(gca,'XTickLabel',VarNames);
set(gca,'YTickLabel',VarNames);
axis([0 x+1 0 x+1]);
grid;
colorbar;

%% Viewing the classification Tree
% Graphic description of the tree

OptimalTree = fitctree(predic,y,'minleaf',40);
view(OptimalTree,'mode','graph')

%% Naive Bayes
% Use a Naive Bayes Classifier to develop a classification model

% Some of the features exhibit significant correlation, however, its
% unclear whether the correlated features will be selected for our model
baymodel=NaiveBayes.fit(Xtrain, Ytrain, 'Distribution','kernel');
Ypredict=predict(baymodel,Xtest);
display(baymodel)
CMat_bay=confusionmat(Ypredict,Ytest)
disp('Naive Bayes Classifier Confusion Matrix')
loss_SVM=(sum(sum(CMat_bay))-sum(diag(CMat_bay)))/sum(sum(CMat_bay));
disp(['Naive Bayes Classifier Loss is:',num2str(loss_SVM*100),'%'])


%% Generate a bagged decision tree

b1 = TreeBagger(250,X,y,'oobvarimp','on');
oobError(b1, 'mode','ensemble')

%% Show out of Bag Feature Importance using the Tree Bagger

figure
bar(b1.OOBPermutedVarDeltaError);
xlabel('Feature');
ylabel('Out-of-bag feature importance');
title('Feature importance results');
set(gca, 'XTickLabel',VarNames)

%%  Run Treebagger Using Sequential Feature Selection

f = @(X,Y)oobError(TreeBagger(50,X,y,'method','classification','oobpred','on'),'mode','ensemble');
opt = statset('display','iter');
[fs,history] = sequentialfs(f,X,y,'options',opt,'cv','none');

%% Rerun the Bagged decision tree with a test set and a training set

b2 = TreeBagger(250,Xtrain,Ytrain,'oobvarimp','on');
oobError(b2, 'mode','ensemble')

X_Test = X(test(c,1), :);
Y_Test = Y(test(c,1));

%% Use the training classifiers to make Predictions about the test set
[Predicted, Class_Score] = predict(b2,Xtest);
Predicted = str2double(Predicted);
[conf, classorder] = confusionmat(Ytest,Predicted);
conf

% Calculate what percentage of the Confusion Matrix is off diagonal
Error3 =  1 - trace(conf)/sum(conf(:))

%% Data Set Partition for Regression Training and Test Sets

c = cvpartition(y1,'holdout');
Xtrain=predic_all(training(c,1),:);
Xtest=predic_all(test(c,1),:);
Ytrain=y1(training(c,1));
Ytest=y1(test(c,1));

%% Multiple Linear Regression Fit to Predict 
modelLR = LinearModel.fit(Xtrain,Ytrain);
disp(modelLR)

yfitLR = predict(modelLR,Xtest);
plotFitErrors(Ytest(:,1),yfitLR)

%% Neural Network

% Create a Fitting Network
hiddenLayerSize = 20;
net_EGT = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
 
% Train the Network
[net_EGT,~] = train(net_EGT,Xtrain.',Ytrain(:,1).');
 
% Test the Network
yfitNN = net_EGT(Xtest.').';

% R_2

plotFitErrors(Ytest(:,1),yfitNN)

R2 = Rsquared(Ytest(:,1),yfitNN);
disp(['R-sq NNet (EGT) = ', num2str(R2)])

view(net_EGT)

%% SVM Model
SVMmodel=svmtrain(Xtrain,Ytrain)
Ypredict=svmclassify(SVMmodel,Xtest);
CMat_bay=confusionmat(predict,Ytest);
disp('SVM Classifier Confusion Matrix')
loss_SVM=(sum(sum(CMat_bay))-sum(diag(CMat_bay)))/sum(sum(CMat_bay));
disp(['SVM Classifier Loss is:',num2str(loss_SVM*100),'%'])