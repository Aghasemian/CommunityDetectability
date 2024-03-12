function y = dnRk_2(x,P,T,xa,ya,xb,yb) 
    y = P * x;
    y_aux=ya*x;
    y_aux=xa*y_aux;
    y=y+y_aux;
    y_aux=yb*x;
    y_aux=xb*y_aux;
    y=y+y_aux;
end

