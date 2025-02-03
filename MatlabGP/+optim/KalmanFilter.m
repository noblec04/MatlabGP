classdef KalmanFilter
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        mu
        sig2
        iter=0;
    end

    methods
        function obj = KalmanFilter()

        end

        function [obj,mu_out,sig2_out] = step(obj,V)
            
            obj.iter = obj.iter + 1;

            if obj.iter == 1
                obj.mu = mean(V,1);
                obj.sig2 = var(V,[],1);

                mu_out = obj.mu;
                sig2_out = obj.sig2;
                return
            end

            mu_in = mean(V,1);
            sig2_in = var(V,[],1);

            alpha = obj.sig2./(obj.sig2 + sig2_in);

            mu_out = alpha.*mu_in + (1 - alpha).*obj.mu;
            sig2_out = (alpha.^2).*sig2_in + ((1 - alpha).^2).*obj.sig2;

            obj.mu = mu_out;
            obj.sig2 = sig2_out;

        end
    end
end