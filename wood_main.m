
addpath('C:\Users\acer\Desktop\Misaj\Project\Wood Dataset\Train\TRAIN 1_12','C:\Users\acer\Desktop\Misaj\Project\Wood Dataset\Test\TEST 1_12');

imagefiles = dir('C:\Users\acer\Desktop\Misaj\Project\Wood Dataset\Train\TRAIN 1_12\*.jpg');
nfiles = length(imagefiles);    % Number of files found

for i=1:nfiles
    currentfilename = imagefiles(i).name;
	[indx1, indx2] = regexp(currentfilename,'\_[0-9]*\.');
	label_values = currentfilename(indx1+1:indx2-1);
% 	disp(label_values)
	currentimage = imread(currentfilename);
	images{i} = currentimage;
   
    %images{i} = imresize(images{i},[512,512]);
    %images{i} = imadjust(images{i},stretchlim(images{i}));
    %rgb to gray
	gray{i} = rgb2gray(images{i});
   
    
    %GLCM feature extraction
	glcms = graycomatrix(gray{i});

    % Derive Statistics from GLCM
	stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
	Contrast = stats.Contrast;
	Correlation = stats.Correlation;
	Energy = stats.Energy;
	Homogeneity = stats.Homogeneity;
    
	feature_value(i,1) = Contrast;
	feature_value(i,2) = Correlation;
	feature_value(i,3) = Energy;
	feature_value(i,4) = Homogeneity;
	feature_value(i,5) = str2double(label_values);
    
    
    X_TRAIN(i,:) = feature_value(i,1:4);
    Y_TRAIN(i) = feature_value(i,5);
    
    
%     disp(X_TRAIN(:,1:4));
    
%   feat_wood(:) = [Contrast,Correlation,Energy,Homogeneity];
%   disp(feat_wood(:));
%   	disp(feature_value(i,5));
end


%     
% j=1;    
% for i=1:nfiles
% %    disp(feature_value(i,1:5));
%     
%     if (feature_value(i,5)== 1) || (feature_value(i,5)== 2)
%        X_TRAIN(j,:) = feature_value(i,1:4);
%        Y_TRAIN(j) = feature_value(i,5);
%        j = j+1;
%     end
% end
% 
imagefiles1 = dir('C:\Users\acer\Desktop\Misaj\Project\Wood Dataset\Test\TEST 1_12\*.jpg');  
nfiles1 = length(imagefiles1);    % Number of files found

for k=1:nfiles1
    currentfilename1 = imagefiles1(k).name;
	[indx1, indx2] = regexp(currentfilename1,'\_[0-9]*\.');
	label_values1 = currentfilename1(indx1+1:indx2-1);
% 	disp(label_values)
	currentimage1 = imread(currentfilename1);
	images{k} = currentimage1;
   
    %images{i} = imresize(images{i},[512,512]);
    %images{i} = imadjust(images{i},stretchlim(images{i}));
    %rgb to gray
	gray1{k} = rgb2gray(images{k});
   
    
    %GLCM feature extraction
	glcms1 = graycomatrix(gray1{k});

    % Derive Statistics from GLCM
	stats1 = graycoprops(glcms1,'Contrast Correlation Energy Homogeneity');
	Contrast1 = stats1.Contrast;
	Correlation1 = stats1.Correlation;
	Energy1 = stats1.Energy;
	Homogeneity1 = stats1.Homogeneity;
    
	feature_value1(k,1) = Contrast1;
	feature_value1(k,2) = Correlation1;
	feature_value1(k,3) = Energy1;
	feature_value1(k,4) = Homogeneity1;
	feature_value1(k,5) = str2double(label_values1);
        
    X_TEST(k,:) = feature_value1(k,1:4);
    Y_TEST(k) = feature_value1(k,5);
%   feat_wood(:) = [Contrast,Correlation,Energy,Homogeneity];
%   disp(feat_wood(:));
%  	disp(feature_value(k,5));
end

% svm_model = svmtrain(X_TRAIN,Y_TRAIN,'kernel_function' ,'rbf');


% predicted_result = multisvm(X_TRAIN,Y_TRAIN,X_TEST);

KNN_model= fitcknn(X_TRAIN,Y_TRAIN);
predicted_label = predict(KNN_model,X_TEST);
% disp(KNN_model);

       








% t = templateSVM('Standardize',true,'KernelFunction','gaussian');
%  svm_model = fitcecoc(X_TRAIN,Y_TRAIN,'Learners',t,'FitPosterior',true,'Verbose',2);
%  CVMdl = crossval(svm_model);
%  genError = kfoldLoss(CVMdl);
 
%  [Y_TEST,~,~,Posterior] = resubPredict(svm_model,'Verbose',1);
 
% predicted_label = svmclassify(svm_model,X_TEST);

cm = confusionmat(Y_TEST,predicted_label);
TP = cm(1,1);
FN = cm(1,2);
FP = cm(2,1);
TN = cm(2,2);

Accuracy = (TP+TN)/(TP+FP+FN+TN);

disp(Accuracy);




















% for i=1:j
%     disp(X(i,1:4));
%     disp(Y(i));
% end






% disp(X_TEST(:,1:4));
% disp(Y_TEST(:));











% for i=1:j
%     disp(X(i,1:4));
%     disp(Y(i));
% end

% disp(X(:,1:4));
% disp(Y(:));

%X_TEST, Y_TEST = wood_test();


% disp(predicted_label);
% disp(Y_TEST);



% disp(svm_model)


% sv = svm_model.SupportVectors;
% figure
% gscatter(X(:,1),X(:,2),Y)
% hold on
% plot(sv(:,1),sv(:,2),10)
% legend('1','2','Support Vector')
% hold off


% feature_value(1,1:5)
% disp(feature_value(1,1));
% disp(feature_value(1,2));
% disp(feature_value(1,3));
% disp(feature_value(1,4));
% disp(feature_value(1,5));




% This fucntion evaluates the performance of a classification model by 
% calculating the common performance measures: Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
% idx = (Y_TEST()==1);
% p = length(Y_TEST(idx));
% n = length(Y_TEST(~idx));
% N = p+n;
% 
% tp = sum(Y_TEST(idx) == predicted_label(idx));
% tn = sum(Y_TEST(~idx) == predicted_label(~idx));
% fp = n-tn;
% fn = p-tp;
% tp_rate = tp/p;
% tn_rate = tn/n;
% accuracy = (tp+tn)/N;
% sensitivity = tp_rate;
% specificity = tn_rate;
% precision = tp/(tp+fp);
% recall = sensitivity;
% f_measure = 2*((precision*recall)/(precision + recall));
% gmean = sqrt(tp_rate*tn_rate);
% EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
% 






