%{
    Non-Linear Multi-Fidelity Gaussian Process
    
    An auto-regressive (AR(N)) model based on Perdikaris

    The model can be constructed using data from nF fidelities.

    First nF GPs must be conditioned on data from each fidelity. These can
    be passed to this class along with means and kernels to be used for the
    GP models on the Zdi terms. The MF model can then be trained to
    optimize these means and kernels HPs.

%}

classdef NLMFGP
    
    properties
        GPs
        Zd

        X
        Y

        lb_x
        ub_x
    end

    methods

        function obj = NLMFGP(GPs,mean,kernel)
            obj.GPs = GPs;
            obj.Zd = GP(mean,kernel);

        end

        function obj = condition(obj,lb,ub)

            if nargin<2
                obj.lb_x = obj.GPs{1}.lb_x;
                obj.ub_x = obj.GPs{1}.ub_x;
            else
                obj.lb_x = lb;
                obj.ub_x = ub;
            end

            nF = numel(obj.GPs);

            Xall = obj.GPs{1}.X;

            for i = 2:nF
                Xn = obj.GPs{i}.eval(obj.GPs{1}.X);
                Xall = [Xall Xn];
            end

            obj.Zd = obj.Zd.condition(Xall,obj.GPs{1}.Y);

            obj.X = obj.GPs{1}.X;
            obj.Y = obj.GPs{1}.Y;
        end

        function obj = train(obj)
            obj.Zd = obj.Zd.train();
        end

        function obj = train2(obj)
            obj.Zd = obj.Zd.train2();
        end

        function obj = resolve(obj,x,y,f)
            
            nF = numel(obj.GPs);

            for i = nF:-1:f+1
                    obj.GPs{i} = obj.GPs{i}.resolve(x,y{i});                    
            end
            
             obj.GPs{f} = obj.GPs{f}.resolve(x,y{f});

             %%%% FINISH %%%%
             obj.Zd = obj.Zd.resolve(x,y{1});
            
        end

        function [y,sig] = eval(obj,x)
            
            nF = numel(obj.GPs);

            Xall = x;

            for i = 2:nF
                Xn = obj.GPs{i}.eval(x);
                Xall = [Xall Xn];
            end

            [y,sig] = obj.Zd.eval(Xall);

        end

        function L = LOO(obj)
            L = obj.Zd.LOO();
        end

        function [y] = eval_mu(obj,x)

                nF = numel(obj.GPs);

                Xall = x;

                for i = 2:nF
                    Xn = obj.GPs{i}.eval(x);
                    Xall = [Xall Xn];
                end

                y = obj.Zd.eval_mu(Xall);

        end

        function [dy] = eval_grad(obj,x)

                nF = numel(obj.GPs);
                nX = size(x,2);

                Xall = x;

                for i = 2:nF
                    Xn = obj.GPs{i}.eval(x);
                    Xall = [Xall Xn];
                end

                dy = obj.Zd.eval_grad(Xall);

                dy = dy(:,1:nX);

        end

        function [sig] = eval_var(obj,x)
            

                nF = numel(obj.GPs);

                Xall = x;

                for i = 2:nF
                    [Xn,Xns] = obj.GPs{i}.eval(x);
                    Xall = [Xall Xn+2*sqrt(Xns)];
                end

                sig = obj.Zd.eval_var(Xall);    
        end

        function [mu,sig] = fantasy(obj,x,y)
            
                nF = numel(obj.GPs);

                Xall = x;

                for i = 2:nF
                    Xall = [Xall y{i}];
                end

                [mu,sig] = obj.Zd.eval(Xall);

        end

        function [R] = expectedReward(obj,x,f)

            [mus,sigs] = obj.eval(x);

            nF = numel(obj.GPs);

            for i = 2:nF
                if i==f
                    [mui,sigi] = obj.GPs{i}.eval(x);
                    y{i} = mui + sigi.^2;
                else
                    y{i} = obj.GPs{i}.eval(x);
                end
            end

            if f == 1

                Xall = [obj.GPs{1}.X;x];

                for i = 2:nF
                    Xn = [obj.GPs{i}.eval(obj.GPs{1}.X);y{i}];
                    Xall = [Xall Xn];
                end

                [mu1,sig1] = obj.GPs{1}.eval(x);
                obj.Zd = obj.Zd.condition(Xall,[obj.GPs{1}.Y;(mu1+2*sqrt(sig1))]);
            end

            [mu,sig] = obj.fantasy(x,y);

            R = (mus-mu).^2 + sig + sigs;

        end

        function y = sample(obj,x)
            [mu,sig] = obj.eval(x);

            y = normrnd(mu,sqrt(sig));
        end

        function y = samplePrior(obj,x)
            
            nF = numel(obj.GPs);

            y = obj.rho{nF}*obj.GPs{nF}.samplePrior(x) + obj.Zd{nF}.samplePrior(x);

            for i = nF-1:-1:2
                
                y = obj.rho{i}*y + obj.Zd{i}.samplePrior(x);
            end

        end

        function y = samplePosterior(obj,x)
            
            nF = numel(obj.GPs);

            y = obj.rho{nF}*obj.GPs{nF}.samplePosterior(x) + obj.Zd{nF}.samplePosterior(x);

            for i = nF-1:-1:2
                
                y = obj.rho{i}*y + obj.Zd{i}.samplePosterior(x);
            end
            
        end

        function warpedobj = exp(obj)

           warpedobj = warpGP(obj,'exp');

        end

        function warpedobj = cos(obj)

           warpedobj = warpGP(obj,'cos');

        end

        function warpedobj = sin(obj)

           warpedobj = warpGP(obj,'sin');

        end

        function warpedobj = mpower(obj,~)

           warpedobj = warpGP(obj,'square');

        end

    end
end