classdef EQ_NN<kernels.Kernel
    
    properties
        theta
        NN
    end

    methods

        function obj = EQ_NN(NN,scale,theta)
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
            obj.NN = NN;
            obj.kernels{1} = obj;
        end

        function [K] = forward_(obj,x1,x2,theta)

            x1 = obj.NN.forward(x1);
            x2 = obj.NN.forward(x2);

            d = obj.dist(x1./theta,x2./theta);

            K = exp(-d.^2);

        end

        function V = getHPs(obj)
            VNN = obj.NN.getHPs();
            V = cell2mat(obj.thetas);
            
            V = [VNN(:);V(:)]';

        end


        
        function obj = setHPs(obj,V)
            nTNN = numel(obj.NN.getHPs());

            obj.NN = obj.NN.setHPs(V(1:nTNN));

            nT = numel(obj.thetas);

            for i = 1:nT
                nTs(i) = numel(obj.thetas{i});
            end

            obj.thetas = mat2cell(V(nTNN+1:nTNN+sum(nTs)),1,nTs);
        end
        
    end
end