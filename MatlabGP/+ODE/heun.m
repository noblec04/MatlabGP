function [t, y] = heun(odefun, ...  % The ODE function
                       y0, ...      % Initial states
                       ts, ...      % Start time
                       tf, ...      % End time
                       h, ...       % Step time
                       AD ...       % AutoDiff param set
                       )           
t = ts:h:tf;

if isa(y0, 'AutoDiff')
    y{1} = y0;
else
    y0_AD.values = y0;
    y0_AD.derivatives = 0*getderivs(AD);

    y0 = AutoDiff(y0_AD);

    y{1} = y0;
end

for k = 2 : length(t)
    yn = y{k-1};
    tn = t(k-1);
    k1 = h * odefun(tn, yn);
    k2 = h * odefun(tn + h, yn + k1);
    
    y{k} = yn + 1 / 2 * k1 + 1 / 2 * k2;
end
end