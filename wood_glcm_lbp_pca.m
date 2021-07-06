
% 
% addpath('C:\Users\Lenovo\Desktop\TRAIN_1_12','C:\Users\Lenovo\Desktop\TEST_1_12');
% imagefiles = dir('C:\Users\Lenovo\Desktop\TRAIN_1_12\*.jpg');

addpath('C:\Users\Lenovo\Desktop\Misaj\Project\Wood Dataset\Train','C:\Users\Lenovo\Desktop\Misaj\Project\Wood Dataset\Test\Test_60');
imagefiles = dir('C:\Users\Lenovo\Desktop\Misaj\Project\Wood Dataset\Train\*.jpg');

nfiles = length(imagefiles);    % Number of files found

for i=1:nfiles
    currentfilename = imagefiles(i).name;
	[indx1, indx2] = regexp(currentfilename,'\_[0-9]*\.');
	label_values = currentfilename(indx1+1:indx2-1)
% 	disp(label_values)
	currentimage = imread(currentfilename);
	images{i} = currentimage;
   
    %images{i} = imresize(images{i},[512,512]);
    %images{i} = imadjust(images{i},stretchlim(images{i}));
    %rgb to gray
	gray{i} = rgb2gray(images{i});
   
    
    %22 GLCM feature extraction
	GLCM2 = graycomatrix(gray{i},'Offset',[2 0;0 2]);
    stats = GLCM_Features1(GLCM2,0);
    
    autocorrelation(1:2) = reshape((stats.autoc),2,[]);
    contrast(1:2) = reshape((stats.contr),2,[]);
    correlation_m(1:2) = reshape((stats.corrm),2,[]);
    correlation(1:2) = reshape((stats.corrp),2,[]);
    cluster_prominence(1:2) = reshape((stats.cprom),2,[]);
    cluster_shade(1:2) = reshape((stats.cshad),2,[]);
    dissimilarity(1:2) = reshape((stats.dissi),2,[]);
    energy_m(1:2) = reshape((stats.energ),2,[]);
    entropy(1:2) = reshape((stats.entro),2,[]);
    homogeneity_m(1:2) = reshape((stats.homom),2,[]);
    homogeneity(1:2) = reshape((stats.homop),2,[]);
    max_probability(1:2) = reshape((stats.maxpr),2,[]);
    variance(1:2) = reshape((stats.sosvh),2,[]);
    sum_average(1:2) = reshape((stats.savgh),2,[]);
    sum_variance(1:2) = reshape((stats.svarh),2,[]);     
    sum_entropy(1:2) = reshape((stats.senth),2,[]);
    difference_variance(1:2) = reshape((stats.dvarh),2,[]);
    difference_entropy(1:2) = reshape((stats.denth),2,[]);
    info_measure_correlation1(1:2) = reshape((stats.inf1h),2,[]);
    info_measure_correlation2(1:2) = reshape((stats.inf2h),2,[]);
    inv_diff_normalized(1:2) = reshape((stats.indnc),2,[]);
    inv_diff_moment_normalized(1:2) = reshape((stats.idmnc),2,[]);
    

	feature_value_GLCM_LBP(i,1) = (autocorrelation(1)+autocorrelation(2))/2;
	feature_value_GLCM_LBP(i,2) = (contrast(1)+contrast(2))/2;
    feature_value_GLCM_LBP(i,3) = (correlation_m(1)+correlation_m(2))/2;
    feature_value_GLCM_LBP(i,4) = (correlation(1)+correlation(2))/2;
    feature_value_GLCM_LBP(i,5) = (cluster_prominence(1)+cluster_prominence(2))/2;
    feature_value_GLCM_LBP(i,6) = (cluster_shade(1)+cluster_shade(2))/2;
    feature_value_GLCM_LBP(i,7) = (dissimilarity(1)+dissimilarity(2))/2;
    feature_value_GLCM_LBP(i,8) = (energy_m(1)+energy_m(2))/2;
    feature_value_GLCM_LBP(i,9) = (entropy(1)+entropy(2))/2;
    feature_value_GLCM_LBP(i,10) = (homogeneity_m(1)+homogeneity_m(2))/2;
    feature_value_GLCM_LBP(i,11) = (homogeneity(1)+homogeneity(2))/2;
    feature_value_GLCM_LBP(i,12) = (max_probability(1)+max_probability(2))/2;
    feature_value_GLCM_LBP(i,13) = (variance(1)+variance(2))/2;
    feature_value_GLCM_LBP(i,14) = (sum_average(1)+sum_average(2))/2;
    feature_value_GLCM_LBP(i,15) = (sum_variance(1)+sum_variance(2))/2;
    feature_value_GLCM_LBP(i,16) = (sum_entropy(1)+sum_entropy(2))/2;
    feature_value_GLCM_LBP(i,17) = (difference_variance(1)+difference_variance(2))/2;
    feature_value_GLCM_LBP(i,18) = (difference_entropy(1)+difference_entropy(2))/2;
    feature_value_GLCM_LBP(i,19) = (info_measure_correlation1(1)+info_measure_correlation1(2))/2;
    feature_value_GLCM_LBP(i,20) = (info_measure_correlation2(1)+info_measure_correlation2(2))/2;
    feature_value_GLCM_LBP(i,21) = (inv_diff_normalized(1)+inv_diff_normalized(2))/2;
    feature_value_GLCM_LBP(i,22) = (inv_diff_moment_normalized(1)+inv_diff_moment_normalized(2))/2;
    
    
    lbpFeatures{i} = extractLBPFeatures(gray{i});
    
    numNeighbors = 8;
    numBins = numNeighbors*(numNeighbors-1)+3;
    feature_value_GLCM_LBP(i,23:81) = reshape((lbpFeatures{i}),numBins,[]);
    
    feature_value_GLCM_LBP(i,82) = str2double(label_values);
    
    X_TRAIN_GLCM_LBP(i,:) = feature_value_GLCM_LBP(i,1:81);
    Y_TRAIN_GLCM_LBP(i) = feature_value_GLCM_LBP(i,82);
end


imagefiles1 = dir('C:\Users\Lenovo\Desktop\Misaj\Project\Wood Dataset\Test\Test_60\*.jpg');  
nfiles1 = length(imagefiles1);    % Number of files found

for k=1:nfiles1
    currentfilename1 = imagefiles1(k).name;
	[indx1, indx2] = regexp(currentfilename1,'\_[0-9]*\.');
	label_values1 = currentfilename1(indx1+1:indx2-1)
% 	disp(label_values)
	currentimage1 = imread(currentfilename1);
	images{k} = currentimage1;
   
    %images{i} = imresize(images{i},[512,512]);
    %images{i} = imadjust(images{i},stretchlim(images{i}));
    %rgb to gray
	gray1{k} = rgb2gray(images{k});
   
    %22 GLCM feature extraction
	GLCM2 = graycomatrix(gray1{k},'Offset',[2 0;0 2]);
    stats1 = GLCM_Features1(GLCM2,0);
    
    autocorrelation(1:2) = reshape((stats1.autoc),2,[]);
    contrast(1:2) = reshape((stats1.contr),2,[]);
    correlation_m(1:2) = reshape((stats1.corrm),2,[]);
    correlation(1:2) = reshape((stats1.corrp),2,[]);
    cluster_prominence(1:2) = reshape((stats1.cprom),2,[]);
    cluster_shade(1:2) = reshape((stats1.cshad),2,[]);
    dissimilarity(1:2) = reshape((stats1.dissi),2,[]);
    energy_m(1:2) = reshape((stats1.energ),2,[]);
    entropy(1:2) = reshape((stats1.entro),2,[]);
    homogeneity_m(1:2) = reshape((stats1.homom),2,[]);
    homogeneity(1:2) = reshape((stats1.homop),2,[]);
    max_probability(1:2) = reshape((stats1.maxpr),2,[]);
    variance(1:2) = reshape((stats1.sosvh),2,[]);
    sum_average(1:2) = reshape((stats1.savgh),2,[]);
    sum_variance(1:2) = reshape((stats1.svarh),2,[]);     
    sum_entropy(1:2) = reshape((stats1.senth),2,[]);
    difference_variance(1:2) = reshape((stats1.dvarh),2,[]);
    difference_entropy(1:2) = reshape((stats1.denth),2,[]);
    info_measure_correlation1(1:2) = reshape((stats1.inf1h),2,[]);
    info_measure_correlation2(1:2) = reshape((stats1.inf2h),2,[]);
    inv_diff_normalized(1:2) = reshape((stats1.indnc),2,[]);
    inv_diff_moment_normalized(1:2) = reshape((stats1.idmnc),2,[]);
    

	feature_value1_GLCM_LBP(k,1) = (autocorrelation(1)+autocorrelation(2))/2;
	feature_value1_GLCM_LBP(k,2) = (contrast(1)+contrast(2))/2;
    feature_value1_GLCM_LBP(k,3) = (correlation_m(1)+correlation_m(2))/2;
    feature_value1_GLCM_LBP(k,4) = (correlation(1)+correlation(2))/2;
    feature_value1_GLCM_LBP(k,5) = (cluster_prominence(1)+cluster_prominence(2))/2;
    feature_value1_GLCM_LBP(k,6) = (cluster_shade(1)+cluster_shade(2))/2;
    feature_value1_GLCM_LBP(k,7) = (dissimilarity(1)+dissimilarity(2))/2;
    feature_value1_GLCM_LBP(k,8) = (energy_m(1)+energy_m(2))/2;
    feature_value1_GLCM_LBP(k,9) = (entropy(1)+entropy(2))/2;
    feature_value1_GLCM_LBP(k,10) = (homogeneity_m(1)+homogeneity_m(2))/2;
    feature_value1_GLCM_LBP(k,11) = (homogeneity(1)+homogeneity(2))/2;
    feature_value1_GLCM_LBP(k,12) = (max_probability(1)+max_probability(2))/2;
    feature_value1_GLCM_LBP(k,13) = (variance(1)+variance(2))/2;
    feature_value1_GLCM_LBP(k,14) = (sum_average(1)+sum_average(2))/2;
    feature_value1_GLCM_LBP(k,15) = (sum_variance(1)+sum_variance(2))/2;
    feature_value1_GLCM_LBP(k,16) = (sum_entropy(1)+sum_entropy(2))/2;
    feature_value1_GLCM_LBP(k,17) = (difference_variance(1)+difference_variance(2))/2;
    feature_value1_GLCM_LBP(k,18) = (difference_entropy(1)+difference_entropy(2))/2;
    feature_value1_GLCM_LBP(k,19) = (info_measure_correlation1(1)+info_measure_correlation1(2))/2;
    feature_value1_GLCM_LBP(k,20) = (info_measure_correlation2(1)+info_measure_correlation2(2))/2;
    feature_value1_GLCM_LBP(k,21) = (inv_diff_normalized(1)+inv_diff_normalized(2))/2;
    feature_value1_GLCM_LBP(k,22) = (inv_diff_moment_normalized(1)+inv_diff_moment_normalized(2))/2;
    
    
    lbpFeatures{k} = extractLBPFeatures(gray{k});
    
    numNeighbors = 8;
    numBins = numNeighbors*(numNeighbors-1)+3;
    feature_value1_GLCM_LBP(k,23:81) = reshape((lbpFeatures{k}),numBins,[]);
    
    feature_value1_GLCM_LBP(k,82) = str2double(label_values1);    
    
    
    X_TEST_GLCM_LBP(k,:) = feature_value1_GLCM_LBP(k,1:81);
    Y_TEST_GLCM_LBP(k) = feature_value1_GLCM_LBP(k,82);
end
 
% 
% KNN_model_GLCM_LBP= fitcknn(X_TRAIN_GLCM_LBP,Y_TRAIN_GLCM_LBP,'distance','hamming','Standardize',1);
% predicted_label = predict(KNN_model_GLCM_LBP,X_TEST_GLCM_LBP);
% disp(KNN_model_GLCM_LBP);

[coeff, score] = pca(X_TRAIN_GLCM_LBP);
reducedDimension = coeff(:,1);
X_reducedTrainData = X_TRAIN_GLCM_LBP * reducedDimension;
X_reducedTestData = X_TEST_GLCM_LBP * reducedDimension;


KNN_model_GLCM_LBP= fitcknn(X_reducedTrainData,Y_TRAIN_GLCM_LBP,'NumNeighbors',1);
predicted_label = predict(KNN_model_GLCM_LBP,X_reducedTestData);
disp(KNN_model_GLCM_LBP)

cm_GLCM_LBP = confusionmat(Y_TEST_GLCM_LBP,predicted_label);

TP_TN=0;
FP_FN=0;
for class_i=1:12
    for class_j=1:12   
        if class_i==class_j
            TP_TN = TP_TN+cm_GLCM_LBP(class_i,class_j);
        else
            FP_FN = FP_FN+cm_GLCM_LBP(class_i,class_j);
        end
    end
end

TP_TN
FP_FN

Accuracy = (TP_TN)/(TP_TN + FP_FN);
disp(Accuracy)

% coeff = pca(X_TRAIN_GLCM_LBP)
% % for pca_num=1:12
% %     for img_num=1:nfiles
%         pca_feature_value_train=

% KNN_model_GLCM_LBP= fitcknn(X_TRAIN_GLCM_LBP,Y_TRAIN_GLCM_LBP,'NumNeighbors',1);
% predicted_label = predict(KNN_model_GLCM_LBP,X_TEST_GLCM_LBP);
% disp(KNN_model_GLCM_LBP);
% 
% 
% cm_GLCM_LBP = confusionmat(Y_TEST_GLCM_LBP,predicted_label);
% 
% TP_TN=0;
% FP_FN=0;
% for class_i=1:12
%     for class_j=1:12
%         
%         if class_i==class_j
%             TP_TN = TP_TN+cm_GLCM_LBP(class_i,class_j);
%         else
%             FP_FN = FP_FN+cm_GLCM_LBP(class_i,class_j);
%         end
%     end
% end
% 
% 
% Accuracy = (TP_TN)/(TP_TN + FP_FN);
% disp(Accuracy)
