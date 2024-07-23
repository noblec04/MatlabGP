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
        rho
        Zd

        lb_x
        ub_x
    end

    methods

        function obj = NLMFGP(GPs,mean,kernel)
            obj.GPs = GPs;

            nF = numel(GPs);

            obj.lb_x = GPs{1}.lb_x;
            obj.ub_x = GPs{1}.ub_x;

            Xall = GPs{1}.X;

            for i = 2:nF
                Xn = GPs{i}.eval(GPs{1}.X);
                Xall = [Xall Xn];
            end

            obj.Zd = GP(mean,kernel);

            obj.Zd = obj.Zd.condition(Xall,GPs{1}.Y);

            obj.Zd = obj.Zd.train();

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

            y = obj.Zd.eval(Xall);

            if nargout>1
                sig = obj.Zd.eval_var(Xall);
            end

        end

        %%%%% FINISH %%%%%%
        function [y,dy] = eval_mu(obj,x)
            
            if nargout<2
                nF = numel(obj.GPs);

                y = obj.rho{nF}*obj.GPs{nF}.eval(x) + obj.Zd{nF}.eval(x);

                for i = nF-1:-1:2

                    y = obj.rho{i}*y + obj.Zd{i}.eval(x);
                end

            else
                nF = numel(obj.GPs);

                [y1,dy1] = obj.GPs{nF}.eval_mu(x);
                [Zd1,dZd1] = obj.Zd{nF}.eval_mu(x);

                y = obj.rho{nF}*y1 + Zd1;
                dy = obj.rho{nF}*dy1 + dZd1;

                for i = nF-1:-1:2
                    [Zdi,dZdi] = obj.Zd{i}.eval_mu(x);

                    y = obj.rho{i}*y + Zdi;
                    dy = obj.rho{i}*dy + dZdi;
                end
            end

        end

        function [sig,dsig] = eval_var(obj,x)
            
            if nargout<2
                nF = numel(obj.GPs);

                sig = (obj.rho{nF}^2)*obj.GPs{nF}.eval_var(x) + obj.Zd{nF}.eval_var(x);

                for i = nF-1:-1:2

                    sig = (obj.rho{i}^2)*sig + obj.Zd{i}.eval_var(x);
                end

            else
                nF = numel(obj.GPs);

                [sig1,dsig1] = obj.GPs{nF}.eval_var(x);
                [Zd1,dZd1] = obj.Zd{nF}.eval_var(x);

                sig = (obj.rho{nF}^2)*sig1 + Zd1;
                dsig = (obj.rho{nF}^2)*dsig1 + dZd1;

                for i = nF-1:-1:2
                    [Zdi,dZdi] = obj.Zd{i}.eval_var(x);

                    sig = (obj.rho{i}^2)*sig + Zdi;
                    dsig = (obj.rho{i}^2)*dsig + dZdi;
                end

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