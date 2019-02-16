function [ lamrrr ] =  lam_rrr(M,H,N,r)
    %calculation of real log canonical threshold of reduced rank reggression (matrix factorization).
    %param M: number of rows (length of each column) in observed matrices
    %param N: number of columns (length of each row) in observed matrices
    %param H: innder dimmension of learning factorization
    %param r: true inner dimmension i.e. rank
    %Refs.
    %% [Aoyagi, 2005]: Miki Aoyagi. Sumio Watanabe. "Stochastic Complexities of Reduced Rank Regression in Bayesian Estimation", Neural Networks, 2005, No. 18, pp.924-933. 
    if (N+r<=M+H) & (M+r<=N+H) & (H+r<=M+N)
        if mod(M+H+N+r,2)==0
            lamrrr=(-(H+r).^2-M.^2-N.^2+2.*(H+r).*(M+N)+2.*M.*N)/8;
        else
            lamrrr=(1-(H+r).^2-M.^2-N.^2+2.*(H+r).*(M+N)+2.*M.*N)/8;
        end
    elseif (M+H<N+r)
        lamrrr=(H.*M-H.*r+N.*r)/2;
    elseif (N+H<M+r)
        lamrrr=(H.*N-H.*r+M.*r)/2;
    elseif (M+N<H+r)
        lamrrr=M.*N/2;
    end
end

