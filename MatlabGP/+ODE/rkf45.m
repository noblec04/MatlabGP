function [t, y] = rkf45(odefun, ...   % The ODE function
                        y0, ...        % Initial states
                        ts, ...        % Start time
                        tf, ...        % End time
                        h, ...         % Step time
                        TOL,...        % Local error tolerance
                        AD ...         % AutoDiff parameter set
                        )          


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

k = 2;
t = ts;
while t < tf
    yn = y{k-1};
    tn = t(k-1);
    k1 = h * odefun(tn, yn);
    k2 = h * odefun(tn+h/4, yn+k1/4);
    k3 = h * odefun(tn+h*3/8, yn+k1*3/32 + k2*9/32);
    k4 = h * odefun(tn+h*12/13, yn + k1*1932/2197 - k2*7200/2197 + ...
                    k3*7296/2197);
    k5 = h * odefun(tn+h, yn + k1*439/216 - 8*k2 + k3*3680/513 - ...
                    k4*845/4104);
    k6 = h * odefun(tn+h/2, yn - k1*8/27 + k2*2 - k3*3544/2565 + ...
                    k4*1859/4104 - k5*11/40);
    
    Y = yn + 25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5;
    Z = yn + 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + ...
        2/55*k6;   
    
    E = norm(Y-Z); % local error estimation    
    h = 0.9 * h * (TOL/E)^(1/5); 
    
    y{k} = Y;

    if isa(h, 'AutoDiff')
        t(k) = t(k-1) + h.values;
    else
        t(k) = t(k-1) + h;
    end
    k = k + 1;
end

end