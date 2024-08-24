classdef MFNN

    properties
        GPs
        MF_NN

        lb_x
        ub_x
    end

    methods

        function obj = MFNN(NNs,MF_NN)
            obj.GPs = NNs;
            obj.MF_NN = MF_NN;
        end

        function [y,obj] = predict(obj,x)

            nF = numel(obj.GPs);

            Xall = x;

            for i = 2:nF
                Xn = obj.GPs{i}.eval_mu(x);
                Xall = [Xall Xn];
            end

            [y] = obj.MF_NN.eval_mu(Xall);

        end

        function mu = eval_mu(obj,x)
            mu = obj.predict(x);
        end

        function sig = eval_var(~,x)
            sig = 0*x(:,1);
        end

        function [mu,sig] = eval(obj,x)
            mu = obj.eval_mu(x);
            sig = obj.eval_var(x);
        end

        function [obj,fval] = train(obj,GPs,x,y,lb,ub)%,xv,fv

            if nargin<4
                obj.lb_x = min(x{end});
                obj.ub_x = max(x{end});
            else
                obj.lb_x = lb;
                obj.ub_x = ub;
            end
            
            obj.GPs = GPs;

            nF = numel(obj.GPs);

            % for i = 1:nF
            %     obj.GPs{i} = obj.GPs{i}.train(x{i},y{i});
            % end

            Xall = x{1};

            for i = 2:nF
                Xn = obj.GPs{i}.eval_mu(x{1});
                Xall = [Xall Xn];
            end

            [obj.MF_NN,fval] = obj.MF_NN.train(Xall,y{1});


        end
    end
end