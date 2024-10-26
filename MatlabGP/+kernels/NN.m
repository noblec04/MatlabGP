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

            x1 = x1./theta;
            x2 = x2./theta;
           
            K = abs(asin(x1*x2'./(sqrt((1 + sqrt(sum(x1.^2,2)))*(1+sqrt(sum(x2.^2,2)))'))));
        end

    end
end