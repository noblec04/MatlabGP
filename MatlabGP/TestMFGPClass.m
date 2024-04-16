clear all
close all
clc

A = 0.5; B = 10; C = -5;
ff{1} = @(x) (6*x-2).^2.*sin(12*x-4);
ff{2} = @(x) 0.4*ff{1}(x)-x-1;
ff{3} = @(x) A*ff{1}(x)+B*(x-0.5)-C; 

xx = linspace(0,1,100)';
yy = ff{1}(xx);


x{1} = [0; lhsdesign(2,1);1];
y{1} = ff{1}(x{1});

for i = 2:3
x{i} = [x{i-1}; lhsdesign(8,1)];
y{i} = ff{i}(x{i});
end


a = Matern52(1,0.2)+RQ(0.1,0.5);
b = EQ(1,0.3);


%%
for i = 1:3
    Z{i} = GP([],a);
    Z{i} = Z{i}|[x{i} y{i}];
    Z{i} = Z{i}.train;
end

%%

MF = MFGP(Z,[],b);

%%

[ys,sig] = Z{1}.eval(xx);

figure(2)
clf(2)
plot(xx,ys)
hold on
plot(xx,ys+2*sqrt(sig),'--')
plot(xx,ys-2*sqrt(sig),'--')
plot(xx,yy,'-.')
plot(x{1},y{1},'x')

%%

[ys,sig] = MF.eval(xx);

figure(3)
clf(3)
plot(xx,ys)
hold on
plot(xx,ys+2*sqrt(sig),'--')
plot(xx,ys-2*sqrt(sig),'--')
plot(xx,yy,'-.')
plot(x{1},y{1},'x')

