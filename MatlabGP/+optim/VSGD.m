classdef VSGD

    %{
    Variational Stochastic Gradient Decent
    Based on the paper:
    @article{
        chen2024variational,
        title={Variational Stochastic Gradient Descent for Deep Neural Networks},
        author={Chen, Haotian and Kuzina, Anna and Esmaeili, Babak and Tomczak, Jakub},
        year={2024},
    }

    input:
    F - anonymous function to minimize (must return value and gradient)
    x0 - initial guess point
    
    Optional Input:
    lb - lower bound (reflective lower bound has been added)
    ub - upper bound (reflective upper bound has been added)
    gamma - prior strength (belief about noise in gradients)
    Kg - variance ratio
    kappa1 - dist learning rate decay 1
    kappa2 - dist learning rate decay 2
    lr - learning rate
    iters - maximum number of iterations
    tol - target tolerance on minimum

    Output:
    x - optimum location
    Fx - value at optimum
    xv - trajectory 
    fv - value at trajectory locations

%}

    properties

        iter = 0;

        lb = [];
        ub = [];

        Kg
        gamma
        kappa1
        kappa2
        
        lr

        ag
        agh

        bg
        bgh
        mug

        fv
        xv
    end

    methods

        function obj = VSGD(x0,varargin)

            input=inputParser;
            input.KeepUnmatched=true;
            input.PartialMatching=false;
            input.addOptional('lb',[]);
            input.addOptional('ub',[]);
            input.addOptional('gamma',1*10^(-7));
            input.addOptional('Kg',30);
            input.addOptional('kappa1',0.81);
            input.addOptional('kappa2',0.9);
            input.addOptional('lr',0.1);
            input.parse(varargin{:})
            in=input.Results;

            obj.lb = in.lb;
            obj.ub = in.ub;

            obj.gamma = in.gamma;
            obj.Kg = in.Kg;
            obj.kappa1 = in.kappa1;
            obj.kappa2 = in.kappa2;

            obj.lr = in.lr;

            obj.ag = 0*x0 + obj.gamma;
            obj.agh = 0*x0 + obj.gamma;

            obj.bg = 0*x0 + obj.gamma;
            obj.bgh = 0*x0 + obj.Kg*obj.gamma;
            obj.mug = 0*x0;
        end

        function [obj,x] = step(obj,x,dF)

            obj.iter = obj.iter + 1;

            rt1 = obj.iter^(-1*obj.kappa1);
            rt2 = obj.iter^(-1*obj.kappa2);

            mugp1 = (obj.bgh./(obj.bgh + obj.bg)).*obj.mug + (obj.bg./(obj.bgh + obj.bg)).*dF;

            sigg2 = 1./((obj.ag./obj.bg) + (obj.agh./obj.bgh));

            obj.ag = obj.gamma+0.5;
            obj.agh = obj.gamma+0.5;

            bgp = obj.gamma + 0.5*(sigg2 + (mugp1 - obj.mug).^2);
            bghp = obj.Kg*obj.gamma + 0.5*(sigg2 + (mugp1 - dF).^2);


            obj.bg = (1-rt1).*obj.bg + rt1*bgp;
            obj.bgh = (1-rt2).*obj.bgh + rt2*bghp;

            obj.mug = mugp1;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%% End Update Algorithm params %%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %update parameters
            x = x - obj.lr*obj.mug./sqrt(obj.mug.^2 + sigg2);

            %reflective upper bound
            if ~isempty(obj.ub)
                for jj = 1:length(x)
                    if x(jj)>obj.ub(jj)
                        x(jj)=obj.ub(jj) - 0.1*abs(obj.lr*obj.mug(jj)./sqrt(obj.mug(jj).^2 + sigg2(jj)));
                    end
                end
            end

            %reflective lower bound
            if ~isempty(obj.lb)
                for jj = 1:length(x)
                    if x(jj)<obj.lb(jj)
                        x(jj)=obj.lb(jj) + 0.1*abs(obj.lr*obj.mug(jj)./sqrt(obj.mug(jj).^2 + sigg2(jj)));
                    end
                end
            end

        end

    end
end