function [y] = StressedPlate(x,i)

T = 25000;
E = 210*10^9;
nu = 0.3;
rho = 7800;

switch i
    case 1
        y = sqrt(1./(rho*x(:,3))).*sqrt(((E*x(:,3).^3)./(12*(1-nu))).*((pi^2)./x(:,1).^2 + (pi^2)./x(:,2).^2).^2 + T*((pi^2)./x(:,1).^2 + (pi^2)./x(:,2).^2)) - 120;
    case 2
        y = sqrt(1./(rho*x(:,3))).*sqrt(((E*x(:,3).^3)./(12*(1-nu))).*((pi^2)./x(:,1).^2 + (pi^2)./x(:,2).^2).^2) - 120;
end


end