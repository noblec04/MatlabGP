function isoplotclasses(kr,dim1,dim2,varargin)

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
        
        Yj = kr.eval([XX;XX]);
        YY(i,j,:) = exp(Yj(1,:))/sum(Yj(1,:),2);        
    end
end

cmap = utils.cmocean('thermal',size(YY,3));

for i = 1:size(YY,3)
    contour(xx2,xx1,YY(:,:,i),[0.5 0.5],'color',cmap(i,:));
    hold on
end

end
