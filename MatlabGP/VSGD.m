function [x,Fx,xv,fv] = VSGD(F,x0,varargin)

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

input=inputParser;
input.KeepUnmatched=true;
input.PartialMatching=false;
input.addOptional('lb',[]);
input.addOptional('ub',[]);
input.addOptional('gamma',1*10^(-7));
input.addOptional('Kg',10);
input.addOptional('kappa1',0.81);
input.addOptional('kappa2',0.9);
input.addOptional('lr',0.1);
input.addOptional('iters',150);
input.addOptional('tol',1*10^(-3));

input.parse(varargin{:})
in=input.Results;

Kg = in.Kg;
gamma = in.gamma;
kappa1 = in.kappa1;
kappa2 = in.kappa2;
lr = in.lr;

ag = gamma;
agh = gamma;

bg = gamma;
bgh = Kg*gamma;
mug = 0*x0;

x = x0;

xv(1,:) = x;

err = 2*in.tol;

i = 0;

while abs(err)>in.tol
    
    i = i+1;

    %evaluate function and gradient
    [Fx,dF] = F(x);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%% Update Algorithm params %%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    rt1 = i^(-1*kappa1);
    rt2 = i^(-1*kappa2);

    mugp1 = (bgh./(bgh + bg)).*mug + (bg./(bgh + bg)).*dF;

    sigg2 = 1./((ag./bg) + (agh./bgh));

    ag = gamma+0.5;
    agh = gamma+0.5;

    bgp = gamma + 0.5*(sigg2 + (mugp1 - mug).^2);
    bghp = Kg*gamma + 0.5*(sigg2 + (mugp1 - dF).^2);


    bg = (1-rt1).*bg + rt1*bgp;
    bgh = (1-rt2).*bgh + rt2*bghp;

    mug = mugp1;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%% End Update Algorithm params %%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %store trajectory
    xv(i,:) = x;
    fv(i,:) = Fx;

    %calculate relative change in min function value
    if i >1
        err = abs(fv(i) - fv(i-1))/(abs(fv(i)) + abs(fv(i-1)));
    end
    
    %update parameters
    x = x - lr*mug./sqrt(mug.^2 + sigg2);

    %reflective upper bound
    if ~isempty(in.ub)
        for jj = 1:length(x)
            if x(jj)>in.ub(jj)
                x(jj)=in.ub(jj) - 0.1*abs(lr*mug(jj)./sqrt(mug(jj).^2 + sigg2(jj)));
            end
        end
    end
    
    %reflective lower bound
    if ~isempty(in.lb)
        for jj = 1:length(x)
            if x(jj)<in.lb(jj)
                x(jj)=in.lb(jj) + 0.1*abs(lr*mug(jj)./sqrt(mug(jj).^2 + sigg2(jj)));
            end
        end
    end

    %if max iterations reached
    if i>in.iters
        break
    end

    %If earlier iteration found smaller function value, shift back
    if Fx>min(fv) && rand()>0.8 && i>2

        [~,ind] = min(fv);
        x = xv(ind,:);
    end

    
    
end


end