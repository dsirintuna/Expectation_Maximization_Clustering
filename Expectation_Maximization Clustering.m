clc,close all,clear all
%% Data Generation
MU1=[2.5 2.5];
SIGMA1=[0.8 -0.6
    -0.6 0.8];
MU2=[-2.5 2.5];
SIGMA2=[0.8 0.6
    0.6 0.8];
MU3=[-2.5 -2.5];
SIGMA3=[0.8 -0.6
    -0.6 0.8];
MU4=[2.5 -2.5];
SIGMA4=[0.8 0.6
    0.6 0.8];
MU5=[0 0];
SIGMA5=[1.6 0
    0 1.6];
X1 = mvnrnd(MU1,SIGMA1,50);
X2 = mvnrnd(MU2,SIGMA2,50);
X3 = mvnrnd(MU3,SIGMA3,50);
X4 = mvnrnd(MU4,SIGMA4,50);
X5 = mvnrnd(MU5,SIGMA5,100);
X=[X1;X2;X3;X4;X5];

%% Random Centroid Initialization
N=300;
K=5;
r = randperm(300,5);
centroids=zeros(5,2);
for i=1:5
    centroids(i,1)=X(r(i),1);
    centroids(i,2)=X(r(i),2);

end

D=zeros(300,5);


%% K-means Clustering for 2 times
for l=1:2
    for i=1:300
        for j=1:5
            D(i,j) = norm(X(i,:) - centroids(j,:));
        end
    end
    [minValues, minIndices] = min(D,[],2);    
    index1=find(minIndices==1);
    centroids(1,1)=mean(X(index1,1));
    centroids(1,2)=mean(X(index1,2));
    index2=find(minIndices==2);
    centroids(2,1)=mean(X(index2,1));
    centroids(2,2)=mean(X(index2,2));
    index3=find(minIndices==3);
    centroids(3,1)=mean(X(index3,1));
    centroids(3,2)=mean(X(index3,2));
    index4=find(minIndices==4);
    centroids(4,1)=mean(X(index4,1));
    centroids(4,2)=mean(X(index4,2));
    index5=find(minIndices==5);
    centroids(5,1)=mean(X(index5,1));
    centroids(5,2)=mean(X(index5,2));
end
s1=0;
s2=0;
s3=0;
s4=0;
s5=0;
for i=1:length(index1)
    s1=s1+(transpose(X(index1(i),:)-centroids(1,:))*(X(index1(i),:)-centroids(1,:)));
end
for i=1:length(index2)
    s2=s2+(transpose(X(index2(i),:)-centroids(2,:))*(X(index2(i),:)-centroids(2,:)));
end
for i=1:length(index3)
    s3=s3+(transpose(X(index3(i),:)-centroids(3,:))*(X(index3(i),:)-centroids(3,:)));
end
for i=1:length(index4)
    s4=s4+(transpose(X(index4(i),:)-centroids(4,:))*(X(index4(i),:)-centroids(4,:)));
end
for i=1:length(index5)
    s5=s5+(transpose(X(index5(i),:)-centroids(5,:))*(X(index5(i),:)-centroids(5,:)));
end
s1=s1/length(index1);
s2=s2/length(index2);
s3=s3/length(index3);
s4=s4/length(index4);
s5=s5/length(index5);

%% Expectation-Maximization Algorithm Initialization

h=0.2*ones(300,5);
sigma = cell(1,5);
sigma{1}=s1;
sigma{2}=s2;
sigma{3}=s3;
sigma{4}=s4;
sigma{5}=s5;
s = zeros(2,2);

ck=[length(index1)/300,length(index2)/300,length(index3)/300,length(index4)/300,length(index5)/300];
%% EM Algorithm
for iteration=1:100
for i=1:300
    for k=1:5
        a(k)=(1/sqrt(((2*pi)^2)*det(sigma{k})))*(exp((-1/2)*(X(i,:)-centroids(k,:))*(inv(sigma{k}))*(transpose(X(i,:)-centroids(k,:)))))*ck(k);
    end
    h(i,:)=a/sum(a);
end


centroids=transpose(transpose(transpose(h)*X)./sum(h,1));

for k=1:5
    for i=1:300
    s=s+(h(i,k)*transpose(X(i,:)-centroids(k,:))*(X(i,:)-centroids(k,:)));
    end
    sigma{k}=s./sum(h(:,k));
    s=zeros(2,2);
end

ck=sum(h,1)/300;

end


%% Finding Group Members
[maxValues, maxIndices] = max(h,[],2);    
group1=find(maxIndices==1);
group2=find(maxIndices==2);
group3=find(maxIndices==3);
group4=find(maxIndices==4);
group5=find(maxIndices==5);

%% Visualization
figure(1)
plot(X(group1,1),X(group1,2),'bo')
xlim([-6 6])
ylim([-6 6])
hold on
plot(X(group2,1),X(group2,2),'ro')
hold on
plot(X(group3,1),X(group3,2),'go')
hold on
plot(X(group4,1),X(group4,2),'mo')
hold on
plot(X(group5,1),X(group5,2),'co')


hold on

x = -6:.1:6; 
y = -6:.1:6; 
[A B] = meshgrid(x,y);
Z1 = mvnpdf([A(:) B(:)],MU1,SIGMA1); 
Z1 = reshape(Z1,size(A)); 
contour(A,B,Z1,[0.05,0.05],'k--'), axis equal 
hold on
Z2 = mvnpdf([A(:) B(:)],MU2,SIGMA2); 
Z2 = reshape(Z2,size(A)); 
contour(A,B,Z2,[0.05,0.05],'k--'), axis equal  
hold on
Z3 = mvnpdf([A(:) B(:)],MU3,SIGMA3); 
Z3 = reshape(Z3,size(A)); 
contour(A,B,Z3,[0.05,0.05],'k--'), axis equal 
hold on
Z4 = mvnpdf([A(:) B(:)],MU4,SIGMA4);
Z4 = reshape(Z4,size(A)); 
contour(A,B,Z4,[0.05,0.05],'k--'), axis equal  
hold on
Z5 = mvnpdf([A(:) B(:)],MU5,SIGMA5); 
Z5 = reshape(Z5,size(A)); 
contour(A,B,Z5,[0.05,0.05],'k--'), axis equal  
%%
ZR1 = mvnpdf([A(:) B(:)],centroids(1,:),sigma{1}); 
ZR1 = reshape(ZR1,size(A)); 
contour(A,B,ZR1,[0.05,0.05],'k'), axis equal 
hold on
ZR2 = mvnpdf([A(:) B(:)],centroids(2,:),sigma{2}); 
ZR2 = reshape(ZR2,size(A)); 
contour(A,B,ZR2,[0.05,0.05],'k'), axis equal  
hold on
ZR3 = mvnpdf([A(:) B(:)],centroids(3,:),sigma{3}); 
ZR3 = reshape(ZR3,size(A)); 
contour(A,B,ZR3,[0.05,0.05],'k'), axis equal 
hold on
ZR4 = mvnpdf([A(:) B(:)],centroids(4,:),sigma{4});
ZR4 = reshape(ZR4,size(A)); 
contour(A,B,ZR4,[0.05,0.05],'k'), axis equal  
hold on
ZR5 = mvnpdf([A(:) B(:)],centroids(5,:),sigma{5}); 
ZR5 = reshape(ZR5,size(A)); 
contour(A,B,ZR5,[0.05,0.05],'k'), axis equal  
centroids