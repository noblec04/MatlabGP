%{
    Multi-Fidelity Gaussian Process
    
    An auto-regressive (AR(1)) model based on Kennedy and O'Hagan using the
    recursive formulation of Le Gratiet.

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

        %%%%% FINISH %%%%%%
        function [y,dy] = eval_mu(obj,x)
            
            if nargout<2
                nF = numel(obj.GPs);

                Xall = x;

                for i = 2:nF
                    Xn = obj.GPs{i}.eval(x);
                    Xall = [Xall Xn];
                end

                y = obj.Zd.eval_mu(Xall);

            else

                nF = numel(obj.GPs);

                Xall = x;

                for i = 2:nF
                    Xn = obj.GPs{i}.eval(x);
                    Xall = [Xall Xn];
                end

                [y, dy] = obj.Zd.eval_mu(Xall);

                dy = dy(:,1:end-(nF-1));
            end

        end

        function [sig,dsig] = eval_var(obj,x)
            
            if nargout<2
                nF = numel(obj.GPs);

                Xall = x;

                for i = 2:nF
                    Xn = obj.GPs{i}.eval(x);
                    Xall = [Xall Xn];
                end

                sig = obj.Zd.eval_var(Xall);

            else

                nF = numel(obj.GPs);

                Xall = x;

                for i = 2:nF
                    Xn = obj.GPs{i}.eval(x);
                    Xall = [Xall Xn];
                end

                [sig, dsig] = obj.Zd.eval_var(Xall);

                dsig = dsig(:,1:end-(nF-1));
            end

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