%{
%clear all
data = load('data.mat');
%label = load('label.mat');

imageTrain = data.imageTrain;
imageTestNew = data.imageTestNew;
labelTestNew = data.labelTestNew;
labelTrain = data.labelTrain;
sampletest = im2double(imread('sampletest.png'))*255;
sampletrain = im2double(imread('sampletrain.png'))*255;

%Part 1
x = (reshape(sampletrain,[1,784]))';
y = (reshape(sampletest,[1,784]))';
a = ((inv((x.')*x))*(x.'))*y;

%Part 2 attempt 2
a_matrix = zeros(500,5000);
for i2=1:500
    for k2=1:5000        
        x2 = (reshape(imageTrain(:,:,k2),[1,784]))';
        y2 = (reshape(imageTestNew(:,:,i2),[1,784]))';
        a_matrix(i2,k2) = ((inv((x2.')*x2))*(x2.'))*y2;        
        k2=k2+1;
    end
    i2=i2+1;
end
%}

%{
%Part 2
totals = [sum(labelTrain(:)==0) sum(labelTrain(:)==1) sum(labelTrain(:)==2) sum(labelTrain(:)==3) sum(labelTrain(:)==4) sum(labelTrain(:)==5) sum(labelTrain(:)==6) sum(labelTrain(:)==7) sum(labelTrain(:)==8) sum(labelTrain(:)==9)];

nearest_neighbour = zeros(500,4);
for i=1:500  %Outer loop for 500 test images    
    distance = zeros(5000,2);
    for k=1:5000 %Inner loop computing distance between test image and all 5000 data points
        distance(k,1) = k;        
        sumtemp = (((abs((imageTestNew(:,:,i) - imageTrain(:,:,k)))).^2));
        sumfull = sum(sumtemp(:));
        distance(k,2) = sqrt(sumfull);                
        k=k+1;
    end    
    Minval = min(distance(:,2));
    [mindist,ind] = min(distance(:,2));
    [m,n] = ind2sub(size(distance(:,2)),ind);
        
    nearest_neighbour(i,1) = i; %Index of which of the 500 test images we're looking at
    nearest_neighbour(i,2) = m; %Index of which of the 5000 is closest
    nearest_neighbour(i,3) = mindist; %Distance From Closest Training to Test
    nearest_neighbour(i,4) = labelTrain(m); %Class of nearest neighbour and class that will be assigned
    
    i=i+1;
end

labelFound = nearest_neighbour(:,4);
graph = zeros(10,1);
for b=1:500
    if labelFound(b) ~= labelTestNew(b)
        graph(labelTestNew(b)+1) = graph(labelTestNew(b)+1)+1;
    end
    b=b+1;
end
totals = [sum(labelTestNew(:)==0) sum(labelTestNew(:)==1) sum(labelTestNew(:)==2) sum(labelTestNew(:)==3) sum(labelTestNew(:)==4) sum(labelTestNew(:)==5) sum(labelTestNew(:)==6) sum(labelTestNew(:)==7) sum(labelTestNew(:)==8) sum(labelTestNew(:)==9)];
new_graph = graph./totals';
figure(1)
scatter([0:9],new_graph)
title('Part 1: Error Rate For Each Case')
xlabel('Classes, i=0 to i=9')
ylabel('Error Rates Per Class')

errors=zeros(500,1);
for c=1:500
    if nearest_neighbour(c,4) == labelTestNew(c)
        errors(c,1) = 0;
    else
        errors(c,1) = 1;
    end
    c=c+1;
end
sumerror = sum(errors); 
P_error = sumerror/500; %Total Error Rate 21.2%
%}

%%{
%Part 3
%apply a to imageTrain
imageTrain = a*imageTrain;

nearest_neighbour3 = zeros(500,4);
for i3=1:500  %Outer loop for 500 test images    
    distance3 = zeros(5000,2);
    for k3=1:5000 %Inner loop computing distance between test image and all 5000 data points
        distance3(k3,1) = k3;        
        sumtemp3 = (((abs((imageTestNew(:,:,i3) - a_matrix(i3,k3)*imageTrain(:,:,k3)))).^2));
        sumfull3 = sum(sumtemp3(:));
        distance3(k3,2) = sqrt(sumfull3);                
        k3=k3+1;
    end    
    Minval3 = min(distance3(:,2));
    [mindist3,ind3] = min(distance3(:,2));
    [m3,n3] = ind2sub(size(distance3(:,2)),ind3);
        
    nearest_neighbour3(i3,1) = i3; %Index of which of the 500 test images we're looking at
    nearest_neighbour3(i3,2) = m3; %Index of which of the 5000 is closest
    nearest_neighbour3(i3,3) = mindist3; %Distance From Closest Training to Test
    nearest_neighbour3(i3,4) = labelTrain(m3); %Class of nearest neighbour and class that will be assigned
    
    i3=i3+1;
end

labelFound3 = nearest_neighbour3(:,4);
graph3 = zeros(10,1);
for b3=1:500
    if labelFound3(b3) ~= labelTestNew(b3)
        graph3(labelTestNew(b3)+1) = graph3(labelTestNew(b3)+1)+1;
    end
    b3=b3+1;
end
totals3 = [sum(labelTestNew(:)==0) sum(labelTestNew(:)==1) sum(labelTestNew(:)==2) sum(labelTestNew(:)==3) sum(labelTestNew(:)==4) sum(labelTestNew(:)==5) sum(labelTestNew(:)==6) sum(labelTestNew(:)==7) sum(labelTestNew(:)==8) sum(labelTestNew(:)==9)];
new_graph3 = graph3./totals3';
figure(1)
scatter([0:9],new_graph3)
title('Part 1: Error Rate For Each Case')
xlabel('Classes, i=0 to i=9')
ylabel('Error Rates Per Class')

errors3=zeros(500,1);
for a3=1:500
    if nearest_neighbour3(a3,4) == labelTestNew(a3)
        errors3(a3,1) = 0;
    else
        errors3(a3,1) = 1;
    end
    a3=a3+1;
end
sumerror3 = sum(errors3); %sumerror is 47; 47 total errors out of 500
P_error3 = sumerror3/500; %P_error is equal to .0940, or 9.4%
%}

