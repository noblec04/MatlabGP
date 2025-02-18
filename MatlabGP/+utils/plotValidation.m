function plotValidation(y1,y2)

figure
subplot(2,1,1)
plot(y1,y2,'.')
axis square

subplot(2,1,2)
plot(y1,(y2-y1)/std(y1),'.')
axis square

end