addpath('C:\Users\acer\Desktop\Misaj\Project\Wood Dataset\Train\TRAIN 1_12','C:\Users\acer\Desktop\Misaj\Project\Wood Dataset\Test\TEST 1_12');

%Training Phase
imagefiles = dir('C:\Users\acer\Desktop\Misaj\Project\Wood Dataset\Train\TRAIN 1_12\*.jpg');
nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
    	currentfilename = imagefiles(i).name;
	[indx1, indx2] = regexp(currentfilename,'\_[0-9]*\.');
	label_values = currentfilename(indx1+1:indx2-1);
% 	disp(label_values)
	currentimage = imread(currentfilename);
	images{i} = currentimage;
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
end


%Testing Phase
imagefiles1 = dir('C:\Users\acer\Desktop\Misaj\Project\Wood Dataset\Test\TEST 1_12\*.jpg');  
nfiles1 = length(imagefiles1);    % Number of files found

for k=1:nfiles1
    currentfilename1 = imagefiles1(k).name;
	[indx1, indx2] = regexp(currentfilename1,'\_[0-9]*\.');
	label_values1 = currentfilename1(indx1+1:indx2-1);
% 	disp(label_values)
	currentimage1 = imread(currentfilename1);
	images{k} = currentimage1;
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
end

KNN_model= fitcknn(X_TRAIN,Y_TRAIN);
predicted_label = predict(KNN_model,X_TEST);
% disp(KNN_model);

%Evaluation
cm = confusionmat(Y_TEST,predicted_label);
TP = cm(1,1);
FN = cm(1,2);
FP = cm(2,1);
TN = cm(2,2);
Accuracy = (TP+TN)/(TP+FP+FN+TN);
disp(Accuracy);
