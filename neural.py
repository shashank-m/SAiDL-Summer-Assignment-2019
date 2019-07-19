import numpy as np    
import random
def create_dataset(rows,columns):
    X_list=[]
    y_list=[]
    for i in range(columns):
        l=[random.randint(0,1) for i in range(rows)]
        X_list.append(l)
        if(l[4]==1):
            output=[l[0]^l[2],l[1]^l[3]]
            y_list.append(output)
        else:
            p=[l[0]^l[2],l[1]^l[3]]
            out=[]
            for i in range(2):
                x_comp=~p[i]
                output=int(bin(x_comp)[-1])
                out.append(output)
            y_list.append(out)
          
    X=np.array(X_list).T   
    Y=np.array(y_list).T
    return X,Y
def initialise_parameters(n_x,n_y,n_h):
    w1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    w2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))
    return w1,b1,w2,b2

X,Y=create_dataset(5,50)   

m=X.shape[1]
n_x=X.shape[0]
n_y=Y.shape[0]
n_h=5
epochs=5000
alpha=0.01

w1,b1,w2,b2=initialise_parameters(n_x,n_y,n_h)  
def forward_prop():
    pass
    
def sigmoid(x):
    return(1/(1+np.exp(-x)))
for i in range(epochs):
    z1=np.dot(w1,X)+b1
    a1=np.tanh(z1)
    z2=np.dot(w2,a1)+b2
    a2=sigmoid(z2)
    
    

    

 


     


