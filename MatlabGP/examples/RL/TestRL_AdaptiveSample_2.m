
clear
close all
clc

layers{1} = NN.FF(200,10);
layers{2} = NN.FF(10,5);
layers{3} = NN.FF(5,5);
layers{4} = NN.FF(5,100);

acts{1} = NN.SNAKE(2);
acts{2} = NN.SNAKE(1);
acts{3} = NN.SNAKE(1);

lss = NN.MAE();

Decider = NN.NN(layers,acts,lss);

LearningRate = 1e5;
gamma = 0.95;

for jj = 1:500
    [ins, outs, indices, dRMSEz, RMSEz, RMAEz] = TestSmoothCircleProblem_NN(Decider,0);

    fprintf('Metrics')
    trapz(dRMSEz')
    RMSEz(end)
    RMAEz(end)

    for t = 1:50

        V = Decider.getHPs();
        [~,dy] = Decider.valueAndGrad(V,ins(:,t)');

        dV = (gamma^t)*LearningRate*dRMSEz(t)*dy(indices(t),:);

        Decider.setHPs(V + dV);
    end

end
