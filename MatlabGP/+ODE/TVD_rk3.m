function [t, y] = TVD_rk3(odefun, ...  % The ODE function
                       y0, ...      % Initial states
                       ts, ...      % Start time
                       tf, ...      % End time
                       h, ...       % Step time
                       AD ...       % AutoDiff parameter set
                       )           
t = ts:h:tf;

if nargin==7
    if isa(y0, 'AutoDiff')
        y{1} = y0;
    else
        y0_AD.values = y0;
        y0_AD.derivatives = 0*getderivs(AD);

        y0 = AutoDiff(y0_AD);

        y{1} = y0;
    end
else
    y{1} = y0;
end

for k = 2 : length(t)
    yn = y{k-1};
    tn = t(k-1);
    k1 = yn + h * odefun(tn, yn);
    k2 = (3/4)*yn + (1/4)*k1 + (1/4)*h * odefun(tn, k1);
    k3 = (1/3)*yn + (2/3)*k2 + (2/3)*h * odefun(tn, k2);
    
    y{k} = k3;
end
end