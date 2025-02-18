
clear
close all
clc

%%
load("examples\NN\digits_2.mat")

for i = 1:500
    YY(i,:,:) = squeeze(XTrain(:,:,i));
end

%%

nmodes = 100;

kb = kernels.Matern52(1,5);

%kb = kernels.DECAY(1,1,1,5);

pod1 = utils.KPOD(nmodes,kb);
%pod1 = utils.RPOD(nmodes);
%pod1 = utils.POD(nmodes);
pod1 = pod1.train(YY);

%%
figure
for i = 1:36
    subplot(6,6,i)
    imagesc(squeeze(pod1.Y(i,:,:)))
    axis square
    axis off
end

figure
for i = 1:36
    subplot(6,6,i)
    imagesc(squeeze(pod1.reconstruct(i)))
    axis square
    axis off
end

figure
for i = 1:36
    subplot(6,6,i)
    pod1.plotModes(i-1)
    axis square
    axis off
end

%%

ll = [0.1 0.5 1 2 3 5 10 20 100];


for i = 1:length(ll)

    kb = kernels.Matern52(1,ll(i));

    pod1 = utils.KPOD(nmodes,kb);
    pod1 = pod1.train(YY);

    figure(3)
    subplot(3,length(ll)+1,i)
    pod1.plotModes(1)
    axis off
    subplot(3,length(ll)+1,i+length(ll)+1)
    pod1.plotModes(3)
    axis off
    subplot(3,length(ll)+1,i+2*(length(ll)+1))
    pod1.plotModes(5)
    axis off

    figure(4)
    subplot(3,length(ll)+1,i)
    imagesc(squeeze(pod1.reconstruct(1,5)))
    axis off
    subplot(3,length(ll)+1,i+length(ll)+1)
    imagesc(squeeze(pod1.reconstruct(1,10)))
    axis off
    subplot(3,length(ll)+1,i+2*(length(ll)+1))
    imagesc(squeeze(pod1.reconstruct(1,50)))
    axis off

end

pod1 = utils.POD(nmodes);
pod1 = pod1.train(YY);

figure(3)
subplot(3,length(ll)+1,i+1)
pod1.plotModes(1)
axis off
subplot(3,length(ll)+1,i+length(ll)+1+1)
pod1.plotModes(3)
axis off
subplot(3,length(ll)+1,i+2*(length(ll)+1)+1)
pod1.plotModes(5)
axis off

figure(4)
subplot(3,length(ll)+1,i+1)
imagesc(squeeze(pod1.reconstruct(1,5)))
axis off
subplot(3,length(ll)+1,i+length(ll)+1+1)
imagesc(squeeze(pod1.reconstruct(1,10)))
axis off
subplot(3,length(ll)+1,i+2*(length(ll)+1)+1)
imagesc(squeeze(pod1.reconstruct(1,50)))
axis off