
clear
clc

f1=@(x)x(:,2)-sin(10*x(:,1))/4-0.5;
f2=@(x)x(:,2)-sin(9*x(:,1))/4-0.4;

x{1}=lhsdesign(100,2);
y{1}=f1(x{1});

x{2}=lhsdesign(1000,2);
y{2}=f2(x{2});

ka = kernels.EQ(1,0.3);
kb = kernels.EQ(1,[0.3 0.3 0.5]);

for i = 1:2
    svm{i} = SVM(ka);
    svm{i} = svm{i}.condition(x{i},y{i},[0 0], [1 1]);
    svm{i} = svm{i}.train();
end

MF = MFSVM(svm,kb);
MF = MF.condition();
MF = MF.train();

figure
utils.isoplot(MF,[0 0],2,1);
hold on
utils.isoplot(svm{1},[0 0],2,1);
utils.isoplot(svm{2},[0 0],2,1);
plot(MF.X(MF.Y>0,1),MF.X(MF.Y>0,2),'x')
plot(MF.X(MF.Y<0,1),MF.X(MF.Y<0,2),'+')

%%

lb = [-5 -5];
ub = [5 5];

f=@(x) testFuncs.SmoothCircle(x,2);

x=lb + (ub-lb).*lhsdesign(500,2);
y=f(x)-0.5;

ka = kernels.RQ(5,1,[0.02 0.02]);

svm=SVM(ka);

svm = svm.condition(x,y);

svm = svm.train();

figure(1)
clf(1)
utils.isoplot(svm,[0 0],2,1);
hold on
plot(svm.X(svm.Y>0,1),svm.X(svm.Y>0,2),'x')
plot(svm.X(svm.Y<0,1),svm.X(svm.Y<0,2),'+')