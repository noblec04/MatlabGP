classdef SGD

    %{
    Stochastic Gradient Decent

    input:
    F - anonymous function to minimize (must return value and gradient)
    x0 - initial guess point
    
    Optional Input:
    lb - lower bound (reflective lower bound has been added)
    ub - upper bound (reflective upper bound has been added)
    lr - learning rate
    iters - maximum number of iterations
    tol - target tolerance on minimum

    Output:
    x - optimum location
    Fx - value at optimum

%}

    properties

        iter = 0;

        lb = [];
        ub = [];
        
        lr

        fv
        xv
    end

    methods

        function obj = SGD(x0,varargin)

            input=inputParser;
            input.KeepUnmatched=true;
            input.PartialMatching=false;
            input.addOptional('lb',[]);
            input.addOptional('ub',[]);
            input.addOptional('lr',0.1);
            input.parse(varargin{:})
            in=input.Results;

            obj.lb = in.lb;
            obj.ub = in.ub;

            obj.lr = in.lr;
        end

        function [obj,x] = step(obj,x,dF)

            obj.iter = obj.iter + 1;

            %update parameters
            x = x - obj.lr*dF;

            %reflective upper bound
            if ~isempty(obj.ub)
                for jj = 1:length(x)
                    if x(jj)>obj.ub(jj)
                        x(jj)=obj.ub(jj) - 0.1*abs(obj.lr*dF(jj));
                    end
                end
            end

            %reflective lower bound
            if ~isempty(obj.lb)
                for jj = 1:length(x)
                    if x(jj)<obj.lb(jj)
                        x(jj)=obj.lb(jj) + 0.1*abs(obj.lr*dF(jj));
                    end
                end
            end

        end

    end
end