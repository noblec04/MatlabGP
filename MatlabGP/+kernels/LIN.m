classdef LIN<kernels.Kernel
    
    properties
        theta
    end

    methods

        function obj = LIN(scale,theta)
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
           
            K = x1*x2';
        end

    end
end