function [t, y] = dormandprince(odefun, ...   % The ODE function
                                y0, ...       % Initial states
                                ts, ...       % Start time
                                tf, ...       % End time
                                h, ...        % Step time
                                TOL,...        % Local error tolerance
                                AD ...         % AutoDiff parameter set
                                )

if isa(y0, 'AutoDiff')
    y{1} = y0;
else
    y0_AD.values = y0;
    y0_AD.derivatives = 0*getderivs(AD);

    y0 = AutoDiff(y0_AD);

    y{1} = y0;
end

k = 2;
t = ts;
while t < tf
    yn = y{k-1};
    tn = t(k-1);
    k1 = h * odefun(tn, yn);
    k2 = h * odefun(tn+h/5, yn+k1/5);
    k3 = h * odefun(tn+h*3/10, yn+k1*3/40 + k2*9/40);
    k4 = h * odefun(tn+h*4/5, yn + k1*44/45 - k2*56/15 + k3*32/9);
    k5 = h * odefun(tn+h*8/9, yn + k1*19372/6561 - k2*25360/2187 + ...
                    k3*64448/6561 - k4*212/729);
    k6 = h * odefun(tn+h, yn + k1*9017/3168 - k2*355/33 + ...
                    k3*46732/5247 + k4*49/176 - k5*5103/18656);
    k7 = h * odefun(tn+h, yn + k1*35/384 + k3*500/1113 + k4*125/192 - ...
                    k5*2187/6784 + k6*11/84);
    
    Y = yn + 35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + ...
        11/84*k6;
    Z = yn + 5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - ...
        92097/339200*k5 + 187/2100*k6 + 1/40*k7;
    
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