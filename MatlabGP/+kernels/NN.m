classdef NN<kernels.Kernel

    properties
        theta
    end

    methods

        function obj = NN(scale,theta)
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.kernels{1} = obj;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
        end

        function [K] = forward_(obj,x1,x2,theta)

            x1 = [0*x1(:,1)+1 x1]./theta;
            x2 = [0*x2(:,1)+1 x2]./theta;

            d12 = x1*x2';

            d11 = diag(x1*x1');
            d22 = diag(x2*x2');

            K = (2/pi)*asin((2*d12)./sqrt((1 + 2*d11)*(1 + 2*d22)'));
        end

    end
end