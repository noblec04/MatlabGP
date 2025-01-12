
x = normrnd(zeros(1,10),sqrt(1/10));

F2 = AutoDiff(normrnd(zeros(2,3),sqrt(1/4)));

x2 = utils.conv1D(x,F2(1,:),0);

x3 = utils.conv1D(x2,F2(1,:),0);

x4 = utils.conv1D(x3,F2(1,:),0);

x5 = x4 + utils.conv1D(x,F2(2,:),0);

loss = sum((x5(:)-x(:)).^2,1);