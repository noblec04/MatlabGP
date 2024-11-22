function plotLineOut(kr,X0,dim,varargin)
% Plot a lineout of the resulting kriging
%
% X0 - provide initial point through which the line will be
% plotted
% dim - the dimension along which the line will be plotted


input=inputParser;
input.KeepUnmatched=true;
input.addOptional('new_fig',false,@islogical);  % Create a new figure
input.addOptional('CI',true,@islogical);        % Confidence interval
input.addOptional('lb',kr.lb_x,@isnumeric);     % Lower bound of plot
input.addOptional('ub',kr.ub_x,@isnumeric);     % Upper bound of plot
input.addOptional('line','-');
input.addOptional('color','k');
input.parse(varargin{:})
in=input.Results;

a=in.lb(dim);
b=in.ub(dim);

n = 200;

xx = linspace(a,b,n);
Xn=zeros(n,length(X0));

for ii=1:length(X0)
    for jj = 1:n
        if ii~=dim
            Xn(jj,ii) = X0(ii);
        else
            Xn(jj,ii) = xx(jj);
        end
    end
end

if in.new_fig
    figure
end

[mu,sig]=kr.eval(Xn);

if in.CI    
    
    FAlpha = 0*sig+1;
    
    CI=bsxfun(@plus,mu,bsxfun(@times,sqrt(sig),norminv([0.025 0.975])));
    CI_h=fill([xx';flipud(xx')],[CI(:,1);flipud(CI(:,2))],'k','FaceColor',in.color,'FaceAlpha',0.3,'EdgeColor','none','CData',[FAlpha; flipud(FAlpha)],'HandleVisibility','off');
    colormap('gray')
    hold on
    
else
    CI_h = [];
end

m_h = plot(Xn(:,dim),mu,in.line,'LineWidth',3,'Color',in.color);

end

