%% This function will generate the network snapshots based on the model in Physical Review X 6, 031005 (2016) 
%% (Preprint at arxiv:1506.06179)

function [As,conf_true_time,cin,cout,lambda,stab]=dsbm_gen_network_new(q,n,T,c,eta,epsilon)
pa=1/q*ones(q,1); %probability of each type
cin=q*c/(1+(q-1)*epsilon);
cout=cin*epsilon;
lambda = (cin-cout)/q/c;
stab=c*lambda*lambda*( (1+eta*eta)/(1-eta*eta) -2/T*eta*eta*( 1-eta^(2*T) )/(1-eta*eta)/(1-eta*eta));
stab_limit=c*lambda*lambda*(1+eta)/(1-eta);

TTM=kron(ones(q,1),(1-eta)*pa')+eta*eye(q); %Temporal Transition Matrix
As=zeros(n,n,T);
conf_true_time=zeros(n,T);

pin=cin/n;
pout=cout/n;
pab=[pin pout;pout pin];

for t=1:T
    if t==1
        type_indices=randsrc(n,1,[1:q;pa']);
    else
        type_indices_aux=type_indices;
        for itr1=1:q
            type=find(type_indices_aux==itr1);
            L_type=length(type);
            type_transitions=randsrc(L_type,1,[1:q;TTM(itr1,:)]);
            type_indices(type,1)=type_transitions;
        end
    end
    A=zeros(n,n);
    for q1=1:q
        for q2=q1:q
            A(type_indices==q1,type_indices==q2)=randsrc(sum(type_indices==q1),sum(type_indices==q2),[0,1;1-pab(q1,q2),pab(q1,q2)]);
            if(q1~=q2)
                A(type_indices==q2,type_indices==q1)=A(type_indices==q1,type_indices==q2)';
            end
        end
    end        
    A=triu(A)+triu(A,1)';
    A=A.*(eye(n)==0);
    m=sum(sum(A))/2;
    label=type_indices;
    conf_true_time(:,t)=label;
    list_edges=[];
    A_edge_checker=triu(A);
    for i=1:n
        edge_loc=find(A_edge_checker(i,:)==1)';
        list_edges=[list_edges;[i*ones(length(edge_loc),1) edge_loc]];
    end
    if(m-size(list_edges,1)~=0)
        error('number of edges are not consistent')
    end
    As(:,:,t)=A;
end