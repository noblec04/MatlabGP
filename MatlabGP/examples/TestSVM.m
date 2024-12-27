

f=@(x)x(:,2)-sin(10*x(:,1))/4-0.5;

x=lhsdesign(300,2);
y=f(x);

ka = kernels.Matern12(1,1)*kernels.EQ(1,1);

svm=SVM(ka);

svm = svm.condition(x,y,[0 0], [1 1]);

svm = svm.train();

figure
utils.isoplot(svm,[0 0],2,1);
hold on
plot(svm.X(svm.Y>0,1),svm.X(svm.Y>0,2),'x')
plot(svm.X(svm.Y<0,1),svm.X(svm.Y<0,2),'+')

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