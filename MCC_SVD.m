% Function that implements robust MCC-SVD
% Zero mean variant
% Author: Carlos Loza
% carlos85loza@gmail.com
%%
function U = MCC_SVD(X, eps, mr, Corr_sigma)
% INPUTS:
% X - Input samples
% eps - Tolerance. Stopping criterion
% mr - Number of principal components to be estimated
% Corr_sigma - Kernel width for correntropy estimation
% OUTPUTS:
% U - First mr principal components. Matrix form

% KEY: For zero mean bases, force mu_t to be zero EVERYTIME

fl = 0;
% Initialization
[U_t,~,~] = svds(X,mr);
mu_t = mean(X,2);
mu_t = zeros(size(mu_t));
[~,n] = size(X);
ct_max = 50;            % Maximum number of iterations of HQ optimization

if nargin == 3
    ct = 1;
    while fl == 0
        % Kernel width calculation - Silverman's rule
        X_cent_t = bsxfun(@minus,X,mu_t);
        aux = (X_cent_t - (U_t*U_t')*X_cent_t);
        X_d_t = sum(aux.^2,1);
        sig_e_t = std(X_d_t);
        R_t = iqr(X_d_t);
        Corr_sigma = 1.06*min([sig_e_t R_t/1.34])*n^(-1/5);
    
        % HQ alternating optimizations
        Corr_arg_t = X_d_t;
        p_t = -exp(-Corr_arg_t/(2*Corr_sigma));  
        %mu_t = (1/sum(p_t))*(sum(bsxfun(@times,p_t,X),2));        
        mu_t = zeros(size(mu_t));       
        X_cent_t = bsxfun(@minus,X,mu_t);
        P_M_t = diag(-p_t);
        PCA_param = X_cent_t*P_M_t*(X_cent_t)';
        
        if mean(isnan(PCA_param(:))) ~= 0
            display('NaN Warning')
        end
        
        if mean(isinf(PCA_param(:))) ~= 0
            display('Inf Warning')
        end
        
        [V,D] = eig(PCA_param);
        [~,idx] = sort(diag(D),'descend');
        V = V(:,idx);
        U_t1 = V(:,1:mr);
        
        % Comparison of successive estimations and stopping flags
        norm_diff = norm(abs(U_t) - abs(U_t1));       
        if ct == ct_max
            fl = 1;
        end      
        if norm_diff < eps
            fl = 1;
        end      
        if fl == 1
            U = U_t1;
        else
            U_t = U_t1;
            ct = ct + 1;
        end
    end  
elseif nargin == 4
    % Kernel width provided
    ct = 1;
    while fl == 0
        % HQ alternating optimizations
        X_cent_t = bsxfun(@minus,X,mu_t);
        aux = (X_cent_t - (U_t*U_t')*X_cent_t);
        X_d_t = sum(aux.^2,1);
        Corr_arg_t = X_d_t;
        p_t = -exp(-Corr_arg_t/(2*Corr_sigma));  
        %mu_t = (1/sum(p_t))*(sum(bsxfun(@times,p_t,X),2));        
        mu_t = zeros(size(mu_t));       
        X_cent_t = bsxfun(@minus,X,mu_t);
        P_M_t = diag(-p_t);
        PCA_param = X_cent_t*P_M_t*(X_cent_t)';
        
        if mean(isnan(PCA_param(:))) ~= 0
            display('NaN Warning')
        end
        
        if mean(isinf(PCA_param(:))) ~= 0
            display('Inf Warning')
        end
        
        [V,D] = eig(PCA_param);
        [~,idx] = sort(diag(D),'descend');
        V = V(:,idx);
        U_t1 = V(:,1:mr);
        
        % Comparison of successive estimations and stopping flags
        norm_diff = norm(abs(U_t) - abs(U_t1));       
        if ct == ct_max
            fl = 1;
        end      
        if norm_diff < eps
            fl = 1;
        end      
        if fl == 1
            U = U_t1;
        else
            U_t = U_t1;
            ct = ct + 1;
        end
    end  
    
end
end
