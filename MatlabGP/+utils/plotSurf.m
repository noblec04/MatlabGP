function plotSurf(kr,x0,dim1,dim2,varargin)

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
input.addOptional('nL',20);
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

for i = 1:n
    for j = 1:n
       
        XX = x0;
        XX(dim1) = xx1(i);
        XX(dim2) = xx2(j);
        
        Yj = kr.eval(XX);
        YY(i,j,:) = Yj(1,:);
        
        if in.CI
            
            [~,ej] = kr.eval([XX;XX]);
            ee(i,j,:) = ej(1,:);
        end
        
    end
end
for i = 1:size(YY,3)
    surf(xx2,xx1,YY(:,:,i),'FaceAlpha',0.9,'EdgeColor','none');
    hold on
end
shading interp
utils.cmocean(in.cmap);

if in.CI 
   for i = 1:size(YY,3)
        surf(xx2,xx1,YY(:,:,i) + 2*sqrt(abs(ee(:,:,i))),'FaceAlpha',0.3,'EdgeColor','none','FaceColor',in.color);
        surf(xx2,xx1,YY(:,:,i) - 2*sqrt(abs(ee(:,:,i))),'FaceAlpha',0.3,'EdgeColor','none','FaceColor',in.color);
   end
end

end
