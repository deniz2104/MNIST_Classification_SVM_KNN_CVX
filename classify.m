function [w,b] = classify(X,y,c)

dim=size(X);
m=dim(1); %numarul de exemple
n=dim(2); %numarul de caracteristici
c=1;
cvx_begin quiet
        variable w(n);
        variable b;
        variable e(m);
        minimize((pow_pos(norm(w),2))/2 +c*sum(e));
        subject to
            e>=0; 
            (y.*(( X*w )+b))>= 1-e; 
cvx_end

end

