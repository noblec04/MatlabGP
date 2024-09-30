classdef VAE

    properties
        Encoder
        Decoder
        lossfunc

        lb_x
        ub_x
    end

    methods

        function obj = VAE(Encoder,Decoder,loss)
            obj.Encoder = Encoder;
            obj.Decoder = Decoder;
            obj.lossfunc = loss;
        end

        function [y,zin,obj] = forward(obj,x)

            [zin] = obj.Encoder.forward(x);
            [zout] = obj.sampleLatent(zin);
            [y] = obj.Decoder.forward(zout);

        end

        function z = sampleLatent(~,z)

            nz = size(z,2)/2;
            z = z(:,1:nz) + exp(z(:,nz+1:2*nz)).*normrnd(0,0*[1:nz]+1);

        end

        function [y] = predict(obj,x)

            [y] = obj.Decoder.forward(x);

        end

        function V = getHPs(obj)

            V=obj.Encoder.getHPs()';
            V=[V;obj.Decoder.getHPs()'];
            
        end

        function obj = setHPs(obj,V)

            nE = numel(obj.Encoder.getHPs());
            Vl = V(1:nE);
            obj.Encoder = obj.Encoder.setHPs(Vl);
            Vl = V(nE+1:end);
            obj.Decoder = obj.Decoder.setHPs(Vl);

        end

        function [e,de] = loss(obj,V,x,y)

            nV = length(V(:));

            V = AutoDiff(V(:));

            obj = obj.setHPs(V(:));

            [yp,zp] = obj.forward(x);

            [eout] = obj.lossfunc.forward(y,yp,zp);

            e1 = sum(eout,2);

            e = getvalue(e1);
            de = getderivs(e1);
            de = reshape(full(de),[1 nV]);

        end

        function [obj,fval,xv,fv] = train(obj,x,y)%,xv,fv

            obj.lb_x = min(x);
            obj.ub_x = max(x);

            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            tx0 = (obj.getHPs())';

            func = @(V) obj.loss(V,x,y);


            %opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'MaxFunctionEvaluations',2000,'MaxIterations',2000,'Display','iter');
            %[theta,fval] = fmincon(func,tx0,[],[],[],[],[],[],[],opts);

            [theta,fval,xv,fv] = VSGD(func,tx0,'lr',0.1,'gamma',100,'iters',3000,'tol',1*10^(-7));

            obj = obj.setHPs(theta(:));
        end
    end
end