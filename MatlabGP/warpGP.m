%{
    Gaussian Process
    
    An exact Gaussian Process with Gaussian Likelihood.

    A mean and kernel function can be created from which the GP can be
    generated.

    The GP can then be conditioned on training data.

    The GP can then be trained to optimize the HPs of the mean and kernel
    by finding the mean of the posterior distribution over parameters, or
    by finding the MAP estimate if the number of HPs is large.

%}

classdef warpGP
    
    properties
        GP
        warping

        lb_x
        ub_x
    end

    methods

        function obj = warpGP(GP,warping)

            obj.GP = GP;
            obj.warping = warping;

            obj.lb_x = obj.GP.lb_x;
            obj.ub_x = obj.GP.ub_x;

        end

        function [mu,sig] = eval(obj,x)
            
           [muN,sigN] = obj.GP.eval(x);

           switch obj.warping

               case 'exp'
                    mu = exp(muN + sigN/2);
                    sig = (exp(sigN) - 1).*exp(2*muN + sigN);
               case 'cos'
                    mu = cos(muN);
                    sig = sigN.*sin(muN).^2;
               case 'sin'
                    mu = sin(muN);
                    sig = sigN.*cos(muN).^2;
               case 'square'
                    mu = muN.^2 + sigN;
                    sig = 2*(muN.^2 + 2*sigN);
           end

        end
        
        
        
    end
end