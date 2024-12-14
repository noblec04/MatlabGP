
clear
close all
clc

A = ([3 3 -1 -3 2 -2 2 ...
        3 -3 2 0 0 -4 4 ...
        0 0 0 0 0 1.75 -1.5 ...
        0 0 0 -1.25 1 0 0]);

tic
[tf, yf] = ODE.rkf45(@(t,y) ODE.test.pleiadesRHS(t,y), A', 0, 15, 0.001, 1e-12);
toc


for i = 1:length(tf)
    yfv(:,i) = (yf{i});
    %dyfv(:,:,i) = full(getderivs(yf{i}));
end

%%


figure
plot(yfv(:,1),yfv(:,8),'--',...
     yfv(:,2),yfv(:,9),'--',...
     yfv(:,3),yfv(:,10),'--',...
     yfv(:,4),yfv(:,11),'--',...
     yfv(:,5),yfv(:,12),'--',...
     yfv(:,6),yfv(:,13),'--',...
     yfv(:,7),yfv(:,14),'--')
title('Position of Pleiades Stars, Solved by ODE89')
xlabel('X Position')
ylabel('Y Position')

%%

[nx,ny,~] = size(dyfv);

figure
k=0;
for i = 1:nx
    for j = 1:ny
        k=k+1;
        subplot(nx,ny,k)
        plot(tf,squeeze(dyfv(i,j,:)))
    end
end



