%{
    Multi-fidelity Support Vector Machine.

    A kernel function can be created from which the SVM can be
    generated.

    The SVM can then be conditioned on training data.

%}

classdef MFSVM
    
    properties
        SVMs
        Zd

        X
        Y

        lb_x=0;
        ub_x=1;
    end

    methods

        function obj = MFSVM(SVMs,kernel)
            obj.SVMs = SVMs;
            obj.Zd = SVM(kernel);
        end

        function [y] = eval(obj,x)
            
            nF = numel(obj.SVMs);

            Xall = x;

            for i = 2:nF
                Xn = obj.SVMs{i}.eval(x);
                Xall = [Xall Xn];
            end

            [y] = obj.Zd.eval(Xall);

        end

        function [dy] = eval_grad(obj,x)
            
            [nn,nx] = size(x);

            x = AutoDiff(x);

            y = obj.eval(x);

            dy = reshape(full(getderivs(y)),[nn,nx]);

        end


        function [obj] = condition(obj,lb,ub)

            if nargin<2
                obj.lb_x = obj.SVMs{1}.lb_x;
                obj.ub_x = obj.SVMs{1}.ub_x;
            else
                obj.lb_x = lb;
                obj.ub_x = ub;
            end

            nF = numel(obj.SVMs);

            Xall = obj.SVMs{1}.X;

            for i = 2:nF
                Xn = obj.SVMs{i}.eval(obj.SVMs{1}.X);
                Xall = [Xall Xn];
            end

            obj.Zd = obj.Zd.condition(Xall,obj.SVMs{1}.Y);

            obj.X = obj.SVMs{1}.X;
            obj.Y = obj.SVMs{1}.Y;


        end

        function [obj] = train(obj)

            obj.Zd = obj.Zd.train();

        end

    end
end