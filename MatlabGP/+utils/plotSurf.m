function plotSurf(kr,dim1,dim2,varargin)

%   dim1 - Vary this dimension
%   dim2 - plot along this dimension
%   f - plot this fidelity

input=inputParser;
input.KeepUnmatched=true;
input.addOptional('new_fig',false,@islogical);  % Create a new figure
input.addOptional('lb',kr.lb_x,@isnumeric);     % Lower bound of plot
input.addOptional('ub',kr.ub_x,@isnumeric);     % Upper bound of plot
input.addOptional('cmap','thermal');
input.addOptional('LS','-');
input.addOptional('nL',50);
input.addOptional('mb',[]);
input.addOptional('CI',true);
input.addOptional('color','b');
input.parse(varargin{:})
in=input.Results;

if in.new_fig
    figure
end

a1=in.lb(dim1);
b1=in.ub(dim1);


a2=in.lb(dim2);
b2=in.ub(dim2);

n = in.nL;

xx1 = linspace(a1,b1,n);
xx2 = linspace(a2,b2,n);

if isempty(in.mb)
    mb = in.lb + 0.5*(in.ub - in.lb);
else
    mb = in.mb;
end

for i = 1:n
    for j = 1:n
       
        XX = mb;
        XX(dim1) = xx1(i);
        XX(dim2) = xx2(j);
        
        YY(i,j) = kr.eval(XX);
        
        if in.CI
            [~,ee(i,j)] = kr.eval(XX);
        end
        
    end
end

surf(xx2,xx1,YY,'FaceAlpha',0.9,'EdgeColor','none');
hold on
shading interp
utils.cmocean(in.cmap);

if in.CI 
   
    surf(xx2,xx1,YY + 2*sqrt(abs(ee)),'FaceAlpha',0.3,'EdgeColor','none','FaceColor',in.color);
    surf(xx2,xx1,YY - 2*sqrt(abs(ee)),'FaceAlpha',0.3,'EdgeColor','none','FaceColor',in.color);
end

end
