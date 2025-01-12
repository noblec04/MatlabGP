
x = normrnd(zeros(10,10),sqrt(1/10));

F2 = AutoDiff(normrnd(zeros(6,3),sqrt(1/4)));

x2 = utils.conv2D(x,F2(1:3,:),0);

x3 = utils.conv2D(x2,F2(1:3,:),0);

x4 = utils.conv2D(x3,F2(1:3,:),0);

x5 = x4 + utils.conv2D(x,F2(4:6,:),0);

loss = sum((x5(:)-x(:)).^2,1);