clear
close all
clc

Dvec = 3;

Wvec = [10 30 100];

CMvec = [2 50 1000];

[WM,CM] = ndgrid(Wvec,CMvec);

XX = [WM(:) CM(:)];
for j = 1:10
    for i = 1:size(XX,1)
        [cost{i,j},R2MF{i,j},RMAEMF{i,j},Ri{i,j},Rie{i,j}] = TestRosenbrockProblem_params(Dvec,XX(i,1),XX(i,2));
    end
end
%%

for j = 1:10
    for i = 1:size(XX,1)
        Crate(i,j) = (log(RMAEMF{i,j}(end)) - log(RMAEMF{i,j}(1)))/(log(cost{i,j}(end)) - log(cost{i,j}(1)));
        Cend(i,j) = cost{i,j}(end);
        Rend(i,j) = RMAEMF{i,j}(end);
    end
end