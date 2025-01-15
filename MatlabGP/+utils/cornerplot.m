function [fig,ax]=cornerplot(data, varargin)
%CORNERPLOT Corner plot showing projections of a multidimensional data set.
%
% CORNERPLOT(DATA) plots every 2D projection of a multidimensional data
% set. DATA is an nSamples-by-nDimensions matrix.
%
% CORNERPLOT(DATA,NAMES) prints the names of each dimension. NAMES is a
% cell array of strings of length nDimensions, or an empty cell array.
%
% CORNERPLOT(DATA,NAMES,TRUTHS) indicates reference values on the plots.
% TRUTHS is a vector of length nDimensions. This might be useful, for instance,
% when looking at samples from fitting to synthetic data, where true
% parameter values are known.
%
% CORNERPLOT(DATA,NAMES,TRUTHS,BOUNDS) indicates lower and upper bounds for
% each dimension. BOUNDS is a 2 by nDimensions matrix where the first row is the
% lower bound for each dimension, and the second row is the upper bound.
%
% CORNERPLOT(DATA,NAMES,TRUTHS,BOUNDS,TOP_MARGIN) plots a number, defined by TOP_MARGIN,
% of empty axes at the top of each column. This can be useful for plotting other statistics
% across parameter values (eg, marginal log likelihood).
%
%
% Output:
% FIG is the handle for the figure, and AX is a
% nDimensions-by-nDimensions array of all subplot handles.
%
% Inspired by corner (github.com/dfm/corner.py)
% by Dan Foreman-Mackey (dan.iel.fm).
%
% Requires kde2d
% (mathworks.com/matlabcentral/fileexchange/17204-kernel-density-estimation/content/kde2d.m)
% by Zdravko Botev (web.maths.unsw.edu.au/~zdravkobotev/).
%
% William Adler, January 2015
% Ver 1.0
% will@wtadler.com
if ~exist('kde2d','file')
    error('You must install <a href="http://www.mathworks.com/matlabcentral/fileexchange/17204-kernel-density-estimation/content/kde2d.m">kde2d.m</a> by <a href="http://web.maths.unsw.edu.au/~zdravkobotev/">Zdravko Botev</a>.')
end
if length(size(data)) ~= 2
    error('x must be 2D.')
end
nDims = min(size(data));  % this assumes that you have more samples than dimensions in your data. Hopefully this is a safe assumption!
% make sure columns are the dimensions of the data
if nDims ~= size(data,2)
    data = data';
end
% assign names and truths if given
names = {};
truths = [];
bounds = [];
bounds_supplied = true;
top_margin = 0;
gutter = [.004 .004];
margins = [.1 .01 .12 .01];
if nargin > 1
    names = varargin{1};
    if ~isempty(names) && ~(iscell(names) && length(names) == nDims)
        error('NAMES must be a cell array with length equal to the number of dimensions in your data.')
    end
    if nargin > 2
        truths = varargin{2};
        if ~isempty(truths) && ~(isfloat(truths) && numel(truths) == nDims)
            error('TRUTHS must be a vector with length equal to the number of dimensions in your data.')
        end
        if nargin > 3
            bounds = varargin{3};
            
            if ~isempty(bounds) && ~(isfloat(bounds) && all(size(bounds) == [2 nDims]))
                error('BOUNDS must be a 2-by-nDims matrix.')
            end
            if nargin > 4
                top_margin = varargin{4};
            end
        end
    end
end
if isempty(bounds) | all(bounds==0)
    bounds = nan(2,nDims);
    bounds_supplied = false;
end
% plotting parameters
fig = figure;
set(gcf, 'color', 'w')
ax = nan(nDims+top_margin,nDims);
hist_bins = 5;
lines = 5;
res = 2^4; % defines grid for which kde2d will compute density. must be a power of 2.
linewidth = 1;
axes_defaults = struct('tickdirmode','manual',...
    'tickdir','out',...
    'ticklength',[.035 .035],...
    'box','off',...
    'xticklabel',[],...
    'yticklabel',[]);
% plot histograms
for i = 1:nDims
    if ~bounds_supplied
        bounds(:,i) = [min(data(:,i)) max(data(:,i))];
    end
    
    for t = 1:top_margin
        ax(i-1+t,i) = tight_subplot(nDims+top_margin, nDims, i-1+t, i, gutter, margins);
        set(gca,'visible','off','xlim',bounds(:,i));
    end
    truncated_data = data;
    truncated_data(truncated_data(:,i)<bounds(1,i) | truncated_data(:,i)>bounds(2,i),i) = nan;
    
    ax(i+top_margin,i) = tight_subplot(nDims+top_margin, nDims, i+top_margin, i, gutter, margins);
    
    h=histogram(truncated_data(:,i), hist_bins, 'normalization', 'pdf','FaceAlpha',0.3,'EdgeColor','white','FaceColor','k');
    [f,xi] = ksdensity(truncated_data(:,i));
    hold on
    plot(xi,f,'LineWidth',3)
    set(gca,'xlim',bounds(:,i),'ylim', [0 max(h.Values)], axes_defaults,'ytick',[]);
    set(gca,'FontSize',12)
    
    
    if i == nDims
        set(gca,'xticklabelmode','auto')
    end
    
    if ~isempty(truths)
        hold on
        plot([truths(i) truths(i)], [0 1], 'k-', 'linewidth',linewidth)
    end
    
    if ~isempty(names)
        if i == 1
            ylabel(names{i});
        end
        if i == nDims
            xlabel(names{i})
        end
    end
    
end
% plot projections
if nDims > 1
    for d1 = 1:nDims-1 % col
        for d2 = d1+1:nDims % row
            [~, density, X, Y] = kde2d([data(:,d1) data(:,d2)],res,[bounds(1,d1) bounds(1,d2)],[bounds(2,d1) bounds(2,d2)]);
            ax(d2+top_margin,d1) = tight_subplot(nDims+top_margin, nDims, d2+top_margin, d1, gutter, margins);
            %contour(X,Y,density, [max(density(:))*0.9 max(density(:))*0.49 max(density(:))*0.1],'LineWidth',2)
            pcolor(X,Y,density)
            shading flat
            grid on
            box on
            utils.cmocean('thermal')
            set(gca,'xlim',bounds(:,d1),'ylim',bounds(:,d2), axes_defaults);
            set(gca,'FontSize',12)
            
            if ~isempty(truths)
                yl = get(gca,'ylim');
                xl = get(gca,'xlim');
                hold on
                plot(xl, [truths(d2) truths(d2)],'k-', 'linewidth',linewidth)
                plot([truths(d1) truths(d1)], yl,'k-', 'linewidth',linewidth)
            end
            if d1 == 1
                if ~isempty(names)
                    ylabel(names{d2})
                end
                set(gca,'yticklabelmode','auto')
            end
            if d2 == nDims
                if ~isempty(names)
                    xlabel(names{d1})
                end
                set(gca,'xticklabelmode','auto')
            end
        end
        
%         % link axes
        row = ax(1+top_margin+d1,:);
        row = row(~isnan(row));
        row = row(1:d1);
        
        col = ax(:,d1);
        col = col(~isnan(col));
        col = col(1:end);
        
        linkaxes(row, 'y');
        linkaxes(col, 'x');
        
    end
end
end
function h=tight_subplot(m, n, row, col, gutter, margins, varargin)
%TIGHT_SUBPLOT Replacement for SUBPLOT. Easier to specify size of grid, row/col, gutter, and margins
%
% TIGHT_SUBPLOT(M, N, ROW, COL) places a subplot on an M by N grid, at a
% specified ROW and COL. ROW and COL can also be ranges
%
% TIGHT_SUBPLOT(M, N, ROW, COL, GUTTER=.002) indicates the width of the spacing
% between subplots, in terms of proportion of the figure size. If GUTTER is
% a 2-length vector, the first number specifies the width of the spacing
% between columns, and the second number specifies the width of the spacing
% between rows. If GUTTER is a scalar, it specifies both widths. For
% instance, GUTTER = .05 will make each gutter equal to 5% of the figure
% width or height.
%
% TIGHT_SUBPLOT(M, N, ROW, COL, GUTTER=.002, MARGINS=[.06 .01 .04 .04]) indicates the margin on
% all four sides of the subplots. MARGINS = [LEFT RIGHT BOTTOM TOP]. This
% allows room for titles, labels, etc.
%
% Will Adler 2015
% will@wtadler.com
if nargin<5 || isempty(gutter)
    gutter = [.002, .002]; %horizontal, vertical
end
if length(gutter)==1
    gutter(2)=gutter;
elseif length(gutter) > 2
    error('GUTTER must be of length 1 or 2')
end
if nargin<6 || isempty(margins)
    margins = [.06 .01 .04 .04]; % L R B T
end
Lmargin = margins(1);
Rmargin = margins(2);
Bmargin = margins(3);
Tmargin = margins(4);
unit_height = (1-Bmargin-Tmargin-(m-1)*gutter(2))/m;
height = length(row)*unit_height + (length(row)-1)*gutter(2);
unit_width = (1-Lmargin-Rmargin-(n-1)*gutter(1))/n;
width = length(col)*unit_width + (length(col)-1)*gutter(1);
bottom = (m-max(row))*(unit_height+gutter(2))+Bmargin;
left   = (min(col)-1)*(unit_width +gutter(1))+Lmargin;
pos_vec= [left bottom width height];
h=subplot('Position', pos_vec, varargin{:});
end

function [bandwidth,density,X,Y]=kde2d(data,n,MIN_XY,MAX_XY)
% fast and accurate state-of-the-art
% bivariate kernel density estimator
% with diagonal bandwidth matrix.
% The kernel is assumed to be Gaussian.
% The two bandwidth parameters are
% chosen optimally without ever
% using/assuming a parametric model for the data or any "rules of thumb".
% Unlike many other procedures, this one
% is immune to accuracy failures in the estimation of
% multimodal densities with widely separated modes (see examples).
% INPUTS: data - an N by 2 array with continuous data
%            n - size of the n by n grid over which the density is computed
%                n has to be a power of 2, otherwise n=2^ceil(log2(n));
%                the default value is 2^8;
% MIN_XY,MAX_XY- limits of the bounding box over which the density is computed;
%                the format is:
%                MIN_XY=[lower_Xlim,lower_Ylim]
%                MAX_XY=[upper_Xlim,upper_Ylim].
%                The dafault limits are computed as:
%                MAX=max(data,[],1); MIN=min(data,[],1); Range=MAX-MIN;
%                MAX_XY=MAX+Range/4; MIN_XY=MIN-Range/4;
% OUTPUT: bandwidth - a row vector with the two optimal
%                     bandwidths for a bivaroate Gaussian kernel;
%                     the format is:
%                     bandwidth=[bandwidth_X, bandwidth_Y];
%          density  - an n by n matrix containing the density values over the n by n grid;
%                     density is not computed unless the function is asked for such an output;
%              X,Y  - the meshgrid over which the variable "density" has been computed;
%                     the intended usage is as follows:
%                     surf(X,Y,density)
% Example (simple Gaussian mixture)
% clear all
%   % generate a Gaussian mixture with distant modes
%   data=[randn(500,2);
%       randn(500,1)+3.5, randn(500,1);];
%   % call the routine
%     [bandwidth,density,X,Y]=kde2d(data);
%   % plot the data and the density estimate
%     contour3(X,Y,density,50), hold on
%     plot(data(:,1),data(:,2),'r.','MarkerSize',5)
%
% Example (Gaussian mixture with distant modes):
%
% clear all
%  % generate a Gaussian mixture with distant modes
%  data=[randn(100,1), randn(100,1)/4;
%      randn(100,1)+18, randn(100,1);
%      randn(100,1)+15, randn(100,1)/2-18;];
%  % call the routine
%    [bandwidth,density,X,Y]=kde2d(data);
%  % plot the data and the density estimate
%  surf(X,Y,density,'LineStyle','none'), view([0,60])
%  colormap hot, hold on, alpha(.8)
%  set(gca, 'color', 'blue');
%  plot(data(:,1),data(:,2),'w.','MarkerSize',5)
%
% Example (Sinusoidal density):
%
% clear all
%   X=rand(1000,1); Y=sin(X*10*pi)+randn(size(X))/3; data=[X,Y];
%  % apply routine
%  [bandwidth,density,X,Y]=kde2d(data);
%  % plot the data and the density estimate
%  surf(X,Y,density,'LineStyle','none'), view([0,70])
%  colormap hot, hold on, alpha(.8)
%  set(gca, 'color', 'blue');
%  plot(data(:,1),data(:,2),'w.','MarkerSize',5)
%
%  Reference:
% Kernel density estimation via diffusion
% Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
% Annals of Statistics, Volume 38, Number 5, pages 2916-2957.
global N A2 I
if nargin<2
    n=2^8;
end
n=2^ceil(log2(n)); % round up n to the next power of 2;
N=size(data,1);
if nargin<3
    MAX=max(data,[],1); MIN=min(data,[],1); Range=MAX-MIN;
    MAX_XY=MAX+Range/2; MIN_XY=MIN-Range/2;
end
scaling=MAX_XY-MIN_XY;
if N<=size(data,2)
    error('data has to be an N by 2 array where each row represents a two dimensional observation')
end
transformed_data=(data-repmat(MIN_XY,N,1))./repmat(scaling,N,1);
%bin the data uniformly using regular grid;
initial_data=ndhist(transformed_data,n);
% discrete cosine transform of initial data
a= dct2d(initial_data);
% now compute the optimal bandwidth^2
  I=(0:n-1).^2; A2=a.^2;
 t_star=root(@(t)(t-evolve(t)),N);
p_02=func([0,2],t_star);p_20=func([2,0],t_star); p_11=func([1,1],t_star);
t_y=(p_02^(3/4)/(4*pi*N*p_20^(3/4)*(p_11+sqrt(p_20*p_02))))^(1/3);
t_x=(p_20^(3/4)/(4*pi*N*p_02^(3/4)*(p_11+sqrt(p_20*p_02))))^(1/3);
% smooth the discrete cosine transform of initial data using t_star
a_t=exp(-(0:n-1)'.^2*pi^2*t_x/2)*exp(-(0:n-1).^2*pi^2*t_y/2).*a; 
% now apply the inverse discrete cosine transform
if nargout>1
    density=idct2d(a_t)*(numel(a_t)/prod(scaling));
	density(density<0)=eps; % remove any negative density values
    [X,Y]=meshgrid(MIN_XY(1):scaling(1)/(n-1):MAX_XY(1),MIN_XY(2):scaling(2)/(n-1):MAX_XY(2));
end
bandwidth=sqrt([t_x,t_y]).*scaling; 
end
%#######################################
function  [out,time]=evolve(t)
global N
Sum_func = func([0,2],t) + func([2,0],t) + 2*func([1,1],t);
time=(2*pi*N*Sum_func)^(-1/3);
out=(t-time)/time;
end
%#######################################
function out=func(s,t)
global N
if sum(s)<=4
    Sum_func=func([s(1)+1,s(2)],t)+func([s(1),s(2)+1],t); const=(1+1/2^(sum(s)+1))/3;
    time=(-2*const*K(s(1))*K(s(2))/N/Sum_func)^(1/(2+sum(s)));
    out=psi(s,time);
else
    out=psi(s,t);
end
end
%#######################################
function out=psi(s,Time)
global I A2
% s is a vector
w=exp(-I*pi^2*Time).*[1,.5*ones(1,length(I)-1)];
wx=w.*(I.^s(1));
wy=w.*(I.^s(2));
out=(-1)^sum(s)*(wy*A2*wx')*pi^(2*sum(s));
end
%#######################################
function out=K(s)
out=(-1)^s*prod((1:2:2*s-1))/sqrt(2*pi);
end
%#######################################
function data=dct2d(data)
% computes the 2 dimensional discrete cosine transform of data
% data is an nd cube
[nrows,ncols]= size(data);
if nrows~=ncols
    error('data is not a square array!')
end
% Compute weights to multiply DFT coefficients
w = [1;2*(exp(-i*(1:nrows-1)*pi/(2*nrows))).'];
weight=w(:,ones(1,ncols));
data=dct1d(dct1d(data)')';
    function transform1d=dct1d(x)
        % Re-order the elements of the columns of x
        x = [ x(1:2:end,:); x(end:-2:2,:) ];
        % Multiply FFT by weights:
        transform1d = real(weight.* fft(x));
    end
end
%#######################################
function data = idct2d(data)
% computes the 2 dimensional inverse discrete cosine transform
[nrows,ncols]=size(data);
% Compute wieghts
w = exp(i*(0:nrows-1)*pi/(2*nrows)).';
weights=w(:,ones(1,ncols));
data=idct1d(idct1d(data)');
    function out=idct1d(x)
        y = real(ifft(weights.*x));
        out = zeros(nrows,ncols);
        out(1:2:nrows,:) = y(1:nrows/2,:);
        out(2:2:nrows,:) = y(nrows:-1:nrows/2+1,:);
    end
end
%#######################################
function binned_data=ndhist(data,M)
% this function computes the histogram
% of an n-dimensional data set;
% 'data' is nrows by n columns
% M is the number of bins used in each dimension
% so that 'binned_data' is a hypercube with
% size length equal to M;
[nrows,ncols]=size(data);
bins=zeros(nrows,ncols);
for i=1:ncols
    [dum,bins(:,i)] = histc(data(:,i),[0:1/M:1],1);
    bins(:,i) = min(bins(:,i),M);
end
% Combine the  vectors of 1D bin counts into a grid of nD bin
% counts.
binned_data = accumarray(bins(all(bins>0,2),:),1/nrows,M(ones(1,ncols)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function t=root(f,N)
% try to find smallest root whenever there is more than one
N=50*(N<=50)+1050*(N>=1050)+N*((N<1050)&(N>50));
tol=10^-12+0.01*(N-50)/1000;
flag=0;
while flag==0
    try
        t=fzero(f,[0,tol]);
        flag=1;
    catch
        tol=min(tol*2,.1); % double search interval
    end
    if tol==.1 % if all else fails
        t=fminbnd(@(x)abs(f(x)),0,.1); flag=1;
    end
end
end