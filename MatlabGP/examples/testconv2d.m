
x = normrnd(zeros(10,10),sqrt(1/10));

F2 = AutoDiff(normrnd(zeros(3,3),sqrt(1/4)));

x2 = utils.conv2D(x,F2,0);

x3 = utils.conv2D(x2,F2,0);

x4 = utils.conv2D(x3,F2,0);

x5 = utils.conv2D(x4,F2,0);