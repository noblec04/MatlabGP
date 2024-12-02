function [t, y] = rk38(odefun, ...  % The ODE function
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
    k1 = h * odefun(tn, yn);
    k2 = h * odefun(tn + h / 3, yn + k1 / 3);
    k3 = h * odefun(tn + h * 2 / 3, yn + - k1 / 3 + k2);
    k4 = h * odefun(tn + h, yn + k1 - k2 + k3);
    
    y{k} = yn + 1 / 8 * k1 + 3 / 8 * k2 + 3 / 8 * k3 + 1 / 8 * k4;
end
end