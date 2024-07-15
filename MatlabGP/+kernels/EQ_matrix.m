classdef EQ_matrix<kernels.Kernel
    
    properties
        theta
    end

    methods

        function obj = EQ_matrix(scale,theta)
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.kernels{1} = obj;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
        end

        function [K] = forward_(~,x1,x2,theta)

            n=0;
            for i = 1:size(x1,2)
                for j = i:size(x2,2)
                    n=n+1;
                    M(i,j) = theta(n);
                end
            end
            
            for i = 1:size(x1,2)
                for j = 1:i
                    n=n+1;
                    M(i,j) = -1*M(j,i);
                end
            end

            for i = 1:size(x1,1)
                for j = 1:size(x2,1)
                   d(i,j) = abs(sqrt((x1(i,:) - x2(j,:))*M*(x1(i,:) - x2(j,:))'));
                end
            end

            K = exp(-(d).^2);

        end
        
    end
end