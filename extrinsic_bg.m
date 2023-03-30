function [bgi,bgspec] = extrinsic_bg(x,Kmx,varargin)

% EXTRINSIC_BG   Extrinsic Background Correction for a set of intensities. 
%
% [BGI,BGSPEC] = EXTRINSIC_BG(X,KMX,VARARGIN) locates the least intense set 
% of pixels in the input set x assuming Poisson statistics using either k-means 
% clustering or a Gaussian mixture model. 
% 
% Inputs
% x: 1-D or 2-D array containing intensities to be clustered 
% Kmx: maximum number of clusters to search for (e.g., 100)
%
% Outputs:
% bgi: logical array the same size as x containing false for non-background
% pixels and true for background pixels
% bgspec (optional): outputs an estimated background spectrum if the array of spectra is input
% 
% Optional Inputs (Name/Value pairs in MATLAB style) 
% 'Algorithm': either 'kmeans' or 'gmm'. defaults to 'kmeans'
% 'Outliers': true/false. true runs an outlier detection based on a nearest-neighbor proximity.
% 'Display': prints various information in the MATLAB command window after the completion of an iteration.
% 'PadEdges': integer. Disregards the pixels that are this number of pixels from the edge of an image.
% 'Replicates': integer. Number of replicates to run in a single instance of kmeans or GMM. defaults to 1. 
% 'Maxiter': integer. maximum number of iterations in a single instance of kmeans or GMM. defaults to 100. 
% 'GMMTolerance': double: convergence tolerance in the GMM. defaults to 1e-6.
% 'SearchAlgorithm': 'bisect' or 'increment'. Defaults to 'bisect'.
% 'Spectra': 2-D array with numel(x) rows. Used to compute a background spectrum if desired.
% 'UseParallel': true/false. Defaults to false.
% 'ManualExclude': linear indices of any pixels to manually exclude from the computation.
%
% Last updated May 23, 2022
% J. Nicholas Taylor
% National Institute of Advanced Industrial Science and Technology
% jntaylor@aist.go.jp

% Parse inputs
[algo,runoutliers,display,pad,reps,maxiter,salgo,spectra,gmmtol,...
    useparallel,manexcl] = parseinputs(varargin);
if ~isempty(spectra) && ~isequal(size(spectra,1),numel(x))
    warning(['''Spectra'' must be a matrix with %d rows if specified. '...
        'Not computing a background spectrum.'],numel(x))
    spectra = [];
end

% Pad edges if input
sz = size(x);
x = x(:);
x = round(x);
[r,c] = ind2sub(sz,1:numel(x));
padi = r(:) <= pad | r(:) > max(r)-pad | c(:) <= pad | c(:) > max(c)-pad;

% remove outliers if input
otli = false(size(x));
if runoutliers
    otli = outliers(x);
end
manexcli = false(size(x));
manexcli(manexcl) = true;
excli = padi | otli | manexcli;
xi = x(~excli);

% Initialize command window display 
if display
    fmt = '\n%s\t%s\t%s\t\t%s\n';
    fprintf(fmt,'K','std.','Poiss. std.','next K')
    dispfmt = '\n%d\t%g\t\t%g%s\t\t%d\n';
end

% Initialize number of clusters
iK = [1 Kmx];
switch salgo
    case 'bisect'
        nextK = floor(mean(iK));
    case 'increment'
        nextK = 2;
    otherwise
        warning(['''searchalgorithm'' must be ''bisect'' or ''increment''' ...
             ' if specified. Defaulting to ''bisect''.'])
        nextK = floor(mean(iK));
        salgo = 'bisect';
end

% Run EBC
flag = false;
intflag = false;
Kmxflag = false;
Klist = [];
bglist = {};
cfpoisslist = [];
slist = [];
pslist = [];
while ~flag
    K = nextK;
    switch algo
        case 'kmeans'
            if useparallel
                [ci,c,s] = km_parshell(xi,K,reps,maxiter);
            else
                [ci,c,s] = km_repshell(xi,K,maxiter,reps);
            end
        case 'gmm'
            if useparallel
                [ci,c,s] = gmm_parshell(xi,K,gmmtol,reps);
            else
                [ci,c,s] = gmm_repshell(xi,K,gmmtol,reps);
            end
    end
    bgpx = ci == 1;
    bgi = false(sz);
    bgi(~excli) = bgpx;
    dspoiss = std(normrnd(c(1),sqrt(c(1)),sum(bgpx),1e4),[],1);
    cfpoiss = sum(dspoiss <= s(1))/numel(dspoiss);
    switch salgo
        case 'bisect'
            Klist = [Klist K];
            bglist = [bglist {bgi}];
            cfpoisslist = [cfpoisslist cfpoiss];
            slist = [slist round(s(1))];
            pslist = [pslist round(mean(dspoiss))];
            if cfpoiss > 0 && cfpoiss < 1
                flag = true;
            end
            if ~intflag
                [nextK,iK,intflag] = nextK_bisect(cfpoiss,iK,K);
            elseif all(dspoiss < s(1)) && abs(K-Kmx) < 1
                flag = true;
                Kmxflag = true;
            else
                ism = ismember(iK,Klist);
                if all(ism)
                    ki = slist;
                    ki(ki > pslist) = 0;
                    [~,ki] = max(ki);
%                     ki = find(Klist == min(iK));
                    bgi = bglist{ki};
                    K = Klist(ki);
                    flag = true;
                else
                    nextK = iK(~ism);
                end
            end
        case 'increment'
            if cfpoiss < 1
                flag = true;
            end
            nextK = K + 1;
    end
    if display
        poissint = [' (' num2str(round(min(dspoiss))) ',' ...
            num2str(round(max(dspoiss))) ')'];
        fprintf(dispfmt,K,round(s(1)),round(mean(dspoiss)),poissint,nextK)
    end

end

% Compute EBC spectrum if input
if ~isempty(spectra)
    bg = logical(bgi(:));
    bgspec = mean(spectra(bg,:),1);
else
    bgspec  = {};
end

% Output diaplay
if display
    if Kmxflag
        warning('\n%s%d%s','Didn''t converge at Kmax = ',K,'. Try increasing Kmax.')
    else
        if intflag
        str = 'integer interval';
        else
            str = 'Poisson variance';
        end
        fmt = '\nConverged to %s at K = %d\n';
        fprintf(fmt,str,K)
    end
end

% Input parser
function [algo,runoutliers,display,pad,reps,maxiter,salgo,spectra,...
    gmmtol,useparallel,manexcl] = parseinputs(args)

n = numel(args);
if logical(rem(n,2))
    error('Input arguments must be specified in name/value pairs.')
end
v = reshape(args,2,n/2);
names = v(1,:);
vals = v(2,:);

ialgo = cellfun(@(x) strcmpi(x,'algorithm'),names);
if any(ialgo)
    algo = vals{ialgo};
    if strcmpi(algo,'kmeans')
        algo = 'kmeans';
    elseif strcmpi(algo,'gmm')
        algo = 'gmm';
    else
        warning(['Algorithm must be ''kmeans'' or ''gmm'' if specified. ' ...
            'Defaulting to kmeans.'])
        algo = 'kmeans';
    end
else
    algo = 'kmeans';
end

iotl = cellfun(@(x) strcmpi(x,'outliers'),names);
if any(iotl)
    runoutliers = vals{iotl};
    if ~islogical(runoutliers)
        if runoutliers > -1 && runoutliers < 2
            runoutliers = logical(runoutliers);
        else
            warning(['''Outliers'' must be true/false if specified. ' ...
                'Turning off outlier detection.'])
            runoutliers = false;
        end
    end
else
    runoutliers = true;
end

idsp = cellfun(@(x) strcmpi(x,'display'),names);
if any(idsp)
    display = vals{idsp};
    if ~islogical(display)
        display = false;
        warning('''Display'' must be true/false if specified. Turning off display.')
    end
else
    display = false;
end    

ipad = cellfun(@(x) strcmpi(x,'padedges'),names);
if any(ipad)
    pad = vals{ipad};
    if ~isnumeric(pad) || logical(rem(pad,1))
        pad = 0;
        warning('''PadEdges'' must be a numeric integer if specified. Turning off edge padding.')
    end
else
    pad = 0;
end

irep = cellfun(@(x) strcmpi(x,'replicates'),names);
if any(irep)
    reps = vals{irep};
    if ~isnumeric(reps) || logical(rem(reps,1))
        reps = 1;
        warning(['''Replicates'' must be an integer if specified. ' ...
            'Performing 1 segmentation replicate.'])
    end
else
    reps = 1;
end

iitr = cellfun(@(x) strcmpi(x,'maxiter'),names);
if any(iitr)
    maxiter = vals{iitr};
    if ~isnumeric(maxiter) || logical(rem(maxiter,1))
        maxiter = 100;
        warning('''MaxIter'' must be a numeric integer if specified. Defaulting to 100.')
    end
else
    maxiter = 100;
end

igmt = cellfun(@(x) strcmpi(x,'gmmtolerance'),names);
if any(igmt)
    gmmtol = vals{igmt};
    if ~isnumeric(gmmtol)
        gmmtol = 1;
        warning('''GMMTolerance'' must be a number if specified. Defaulting to 1e-6.')
    end
else
    gmmtol = 1e-6;
end

isalgo = cellfun(@(x) strcmpi(x,'searchalgorithm'),names);
if any(isalgo)
    salgo = vals{isalgo};
    if ~any(strcmpi(salgo,{'bisect','increment'}))
        salgo = 'bisect';
        warning(['SearchAlgorithm must be ''bisect'' or ''increment'''...
            'if specified. Defaulting to ''bisect''.'])
    end
else
    salgo = 'bisect';
end

ispecs = cellfun(@(x) strcmpi(x,'spectra'),names);
if any(ispecs)
    spectra = vals{ispecs};
else
    spectra = [];
end

ipar = cellfun(@(x) strcmpi(x,'useparallel'),names);
if any(ipar)
    useparallel = vals{ipar};
    if ~islogical(useparallel)
        useparallel = false;
        warning(['''UseParallel'' must be true/false if specified. '...
            'Turning off parallel computation.'])
    end
else
    useparallel = false;
end    
if any(ipar)
    v = ver;
    v = {v.Name};
    parcheck = any(strcmpi(v,'Parallel Computing Toolbox'));
    if ~parcheck
        useparallel = false;
        warning(['''UseParallel'' reqires the Parallel Computing Toolbox. '...
            'Turning off parallel computation'])
    end
end

imexcl = cellfun(@(x) strcmpi(x,'manualexclude'),names);
if any(imexcl)
    manexcl = vals{imexcl}(:);
else
    manexcl = [];
end

% proximity-based outlier detection
function otli = outliers(x)

N = numel(x);
[sx,si] = sort(x,'ascend');
nn = 16;%ceil(0.01*numel(xp));
dnn = cell(N,1);
for prox = 1:nn
    b = [1; zeros(prox-1,1); -1];
    ff = sqrt(filter(b,1,sx).^2);
    rf = flipud(sqrt(filter(b,1,flipud(sx)).^2));
    ff = ff(prox+1:end);
    rf = rf(1:end-prox);
    ff = num2cell(ff); ff = [cell(prox,1); ff];
    rf = num2cell(rf); rf = [rf; cell(prox,1)];
    dnn = cellfun(@(x,y,z) cat(2,x,y,z),dnn,ff,rf,'uni',0);
end
dnnt = cellfun(@(x) mean(x(1:nn)),dnn);
dnn = zeros(size(x));
dnn(si) = dnnt;
otli = dnn > quantile(dnn,0.9995);

% shell for running km_local in parallel
function [ci,m,s] = km_parshell(x,K,reps,maxiter)

for k = 1:reps 
    f(k) = parfeval(@km_local,4,x,K,maxiter); 
end
wait(f)
[ci,m,s,d] = fetchOutputs(f,'UniformOutput',false);
[~,idx] = min(cat(1,d{:}));
ci = ci{idx};
m = m{idx};
s = s{idx};

% shell for running km_local without parallel 
function [ci,m,s] = km_repshell(x,K,maxiter,reps)
ci = cell(1,reps);
m = cell(1,reps);
s = cell(1,reps);
d = zeros(1,reps);
for k = 1:reps 
    [ci{k},m{k},s{k},d(k)] = km_local(x,K,maxiter); 
end
[~,idx] = min(d);
ci = ci{idx};
m = m{idx};
s = s{idx};

% local kmeans that doesn't require Stats & ML toolbox
function [i,c,s,sumd] = km_local(x,K,maxiter)

N = numel(x);
c = x(randi(K));
k = 1;
while k < K
    d = min((x - repmat(c,N,1)).^2,[],2);
    p = (d.^2)/sum(d.^2);
    c = [c x(randsample(1:N,1,true,p))];
    k = numel(c);
end
[d,i] = min((x - repmat(c(:)',N,1)).^2,[],2);
i0 = i;
sumd = sum(d);
cflag = false;
iter = 0;
while ~cflag
    iter = iter + 1;
    c = sort(accumarray(i0,x,[K 1],@mean),'ascend');
    [d,i] = min((x - c(:)').^2,[],2);
    empty = ~logical(accumarray(i,ones(numel(i),1),[K 1]));
    if any(empty)
        nempty = sum(empty);
        c(empty) = [];
        [~,di] = sort(d,'descend');
        newc = x(di(1:nempty));
        c = sort([c newc],'ascend');
        [d,i] = min((x - c(:)').^2,[],2);
    end
    sumd = sum(d);
    mv = sum(i0 ~= i);
    if mv < 1
        cflag = true;
        c = sort(accumarray(i,x,[K 1],@mean),'ascend');
    else
        i0 = i;
    end
    if iter >= maxiter
        cflag = true;
        c = sort(accumarray(i,x,[K 1],@mean),'ascend');
    end
end
s = accumarray(i,x,[K 1],@std);


% Choose next number of clusters for bisection search
function [nextK,iK,intflag] = nextK_bisect(cfpoiss,iK,K)

if cfpoiss < 1
    iK = [iK(1) K];
else
    iK = [K iK(2)];
end
intflag = abs(diff(iK)) < 2;
if intflag
    nextK = iK(iK ~= K);
else
    nextK = floor(mean(iK));
end

% Shell for running GMM in parallel
function [ci,m,s] = gmm_parshell(x,K,gmmtol,reps)
for k = 1:reps 
    f(k) = parfeval(@em_gmm,4,x,K,gmmtol); 
end
wait(f)
[ci,m,s,L] = fetchOutputs(f,'UniformOutput',false);
[~,idx] = max(cat(1,L{:}));
ci = ci{idx};
m = m{idx};
s = s{idx};

% Shell for running GMM without parallel
function [ci,m,s] = gmm_repshell(x,K,gmmtol,reps)
ci = cell(1,reps);
m = cell(1,reps);
s = cell(1,reps);
L = zeros(1,reps);
for k = 1:reps 
    [ci{k},m{k},s{k},L(k)] = em_gmm(x,K,gmmtol); 
end
[~,idx] = max(L);
ci = ci{idx};
m = m{idx};
s = s{idx};

% Gaussian mixture model with expectation-maximization optimizer
function [ci,m,s,L] = em_gmm(x,K,tol)

x = x(:)';
N = numel(x);
% initialize parameters
m = x(randperm(N,K));
[~,ci] = min(sqrt((repmat(m(:),1,N)-repmat(x,K,1)).^2));
w = accumarray(ci(:),ones(N,1)/N);
s = accumarray(ci(:),x(:),[],@std);

% initialize likelihood
p = repmat(w(:),1,N).*normpdf(repmat(x,K,1),repmat(m(:),1,N),...
    repmat(s(:),1,N));
L = sum(sum(log(p),1));

% iterate until convergence within tolerance
exit = false;
itr = 0;
Ln = L;
while ~exit
    itr = itr + 1;
    phi = p./repmat(sum(p,1),K,1);                                         % membership: p/sum(p,1)
    nk = sum(phi,2);                                                       % effective number
    w = nk/N;                                                              % update weight
    m = sum(phi.*x./nk,2);                                                 % update mean
    s = sqrt(sum(phi.*((x-repmat(m(:),1,N)).^2)./repmat(nk,1,N),2));       % update std
    p = repmat(w(:),1,N).*normpdf(repmat(x,K,1),repmat(m(:),1,N),...       % update likelihood
        repmat(s(:),1,N));
    L = sum(log(sum(p,1)));                                                % log likelihood
    dL = abs(abs(L-Ln)/L);                                                 % check for convergence
    if dL <= tol
        Ln = L;
        exit = true;
    elseif itr > 1e4
        exit = true;
    else
        Ln = L;
    end
end
[m,i] = sort(m,'ascend');
p = p(i,:);
phi = p./repmat(sum(p,1),K,1);
[~,ci] = max(phi',[],2);
s = accumarray(ci,x(:),[],@std);





