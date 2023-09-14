# -*- coding: utf-8 -*-


import torch as tc
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time

learning_rate = 0.01
iteration_number = 30000
v = 0.1


initial_time = time.time()
print(initial_time)

class Qu4(nn.Module):
    loss_history = [10]

    def __init__(self):
        super(Qu4, self).__init__()

        # Define the parameters of your model here.
        self.params = nn.Parameter(tc.randn(121,dtype=tc.float64),requires_grad=True)


        
        

    def generate_state(self,a):
        a = tc.exp(a)
        
        a = nn.functional.normalize(a,dim=0,eps=1e-14)
        
        a = tc.unsqueeze(a,-1)
        



        psi012 = tc.cat((a[0],tc.zeros(3,device=self.params.device),a[1],a[2], a[3],a[4]))
        psi345 = psi012
        psi678 = psi012
        psi911 = psi012

        psi012345 = tc.kron(psi012,psi345)
        psi012345678 = tc.kron(psi012345,psi678)
        psi012345678911 = tc.kron(psi012345678,psi911)

        return psi012345678911
    
    

    def swap_state(self,state):

        return tc.matmul(swap,state)

    def density_matrix(self,state):
        return tc.outer(state,state.conj())

    def povm(self,density,m1,m2,m3,m4):
        ma = tc.kron(tc.kron(m1,m2),m3)
        ma_ = tc.kron(ma,m4)
        mb = tc.matmul(density,ma_)
        return tc.trace(mb)

    
    def generate_POVM(self,x):

        # Stack them along a new dimension
        stacked_tensors = tc.stack([x[0:8], x[8:16], x[16:24], x[24:32]], dim=0)

        # Apply softmax along this dimension
        eigen = nn.functional.softmax(stacked_tensors, dim=0)
        m = tc.zeros((4, 8, 8), dtype=tc.float64,device=self.params.device) 
        idx = tc.triu_indices(8, 8, offset=1)  # indices for the upper triangular part excluding the diagonal

        for i in range (3):
            D =tc.diag(eigen[i])
            A = tc.zeros((8, 8), dtype=tc.float64,device=self.params.device)
    
            A[idx[0], idx[1]] = x[32+28*i:60+28*i]
            A = A - A.T  # to make A skew-symmetric

            # generate an orthogonal matrix O using the skew-symmetric matrix
            U = tc.linalg.matrix_exp(A)
 
            m[i] = tc.mm(tc.mm(U, D), U.t())
        m[3] = tc.eye(8,device=self.params.device) - m[0]-m[1]-m[2]
        return m
    
    

    def povm_result(self,rou,m):

        a = tc.zeros((4,4,4,4),dtype=tc.float64,device=self.params.device)

        for q in range(4):
            for w in range(4):
                for e in range(4):
                    for r in range(4):
                        a[q,w,e,r] = self.povm(rou,m[q],m[w],m[e],m[r])

        return a





    def obj_fun(self,obj,measure):
        kl= 0
        list_obj = tc.flatten(obj)
        list_m = tc.flatten(measure)

        #pdist =nn.PairwiseDistance(p=2)
        #out = pdist(list_obj,list_m)
        for i in range(256):
            kl += list_obj[i] * tc.log(list_obj[i]/list_m[i])
        return kl

    def loss_KL(self):

        state = self.generate_state(self.params[0:5])

        state_swaped = self.swap_state(state)
        rou = self.density_matrix(state_swaped)

        m = self.generate_POVM(self.params[5:121])
        
        

        result_list = self.povm_result(rou,m)


        y = self.obj_fun(object_list,result_list)

        return y,result_list,m

    def optimize(self,optimizer, epochs=1000):

        for i in range(epochs):
            optimizer.zero_grad()  # clear previous gradients
            loss,result,povm = self.loss_KL()  # compute the loss
            if loss.item()<=1e-12:
                print('Bravo!!!!!!!!!!!!!!!!!!!')

               # tc.save(self.params,'gpu_qu4_v0.36_aprox.pt')



                break
            if abs(loss.item()-Qu4.loss_history[-1])<=1e-12 and abs(loss.item()-Qu4.loss_history[-1])<=1e-3*loss.item():
               break

            loss.backward()  # compute updates for each parameter
            optimizer.step()  # make the updates for each parameter

            # Record the loss
            Qu4.loss_history.append(loss.item())
            if i%10 == 0:
                print(i,loss)
            
        print('Result - Target =',result-object_list)
        print('Equivalent v = ',1-2**8*result[0,0,0,0])
        print('M1 = ',povm[0])
        print('M2 = ',povm[1])
        print('M3 = ',povm[2])
        print('M4 = ',povm[3])


    def draw(self):
        plt.figure(figsize=(10, 6))
        plt.plot(Qu4.loss_history)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over time')
        plt.savefig('plot.png')



def generate_swapmatrix():

    d = 2
    s = tc.zeros(d**2, d**2,dtype=tc.float64,device = 'cuda')

    for i in range(d):
        for j in range(d):
            row = i * d + j
            col = j * d + i
            s[row, col] = 1



    s0123 = tc.kron(tc.eye(d**2,device = 'cuda'),s)

    s01234 = tc.kron(s0123,tc.eye(d,device = 'cuda'))
    s0123456 = tc.kron(s01234,s)
    s01234567 = tc.kron(s0123456,tc.eye(d,device = 'cuda'))
    s0123456789 = tc.kron(s01234567,s)
    s012345678911 = tc.kron(s0123456789,tc.eye(d**2,device = 'cuda'))
    
    s01 = tc.kron(s,tc.eye(d**10,device = 'cuda'))


    s12_ = tc.kron(tc.eye(d,device = 'cuda'),s)
    s12 = tc.kron(s12_,tc.eye(d**9,device = 'cuda'))

    s23_ = tc.kron(tc.eye(d**2,device = 'cuda'),s)
    s23 = tc.kron(s23_,tc.eye(d**8,device = 'cuda'))

    s34_ = tc.kron(tc.eye(d**3,device = 'cuda'),s)
    s34 = tc.kron(s34_,tc.eye(d**7,device = 'cuda'))

    s45_ = tc.kron(tc.eye(d**4,device = 'cuda'),s)
    s45 = tc.kron(s45_,tc.eye(d**6,device = 'cuda'))

    s56_ = tc.kron(tc.eye(d**5,device = 'cuda'),s)
    s56 = tc.kron(s56_,tc.eye(d**5,device = 'cuda'))

    s67_ = tc.kron(tc.eye(d**6,device = 'cuda'),s)
    s67 = tc.kron(s67_,tc.eye(d**4,device = 'cuda'))

    s78_ = tc.kron(tc.eye(d**7,device = 'cuda'),s)
    s78 = tc.kron(s78_,tc.eye(d**3,device = 'cuda'))

    s89_ = tc.kron(tc.eye(d**8,device = 'cuda'),s)
    s89 = tc.kron(s89_,tc.eye(d**2,device = 'cuda'))

    s90_ = tc.kron(tc.eye(d**9,device = 'cuda'),s)
    s90 = tc.kron(s90_,tc.eye(d**1,device = 'cuda'))

    s11 = tc.kron(tc.eye(d**10,device = 'cuda'),s)
    
    

    swap_far = tc.einsum('ab,bc,cd,de,ef,fg,gh,hi,ij,jk,kl,lm,mn,no,op,pq,qr,rs,st,tu,uv->av', s01,s12,s23,s34,s45,s56,s67,s78,s89,s90,s11,s90,s89,s78,s67,s56,s45, s34, s23, s12, s01)


    swap = tc.einsum('ab,bc->ac',swap_far,s012345678911)


    return swap
swap = generate_swapmatrix()
#swap=swap.to('cuda')

def obj_list():
    p = tc.zeros((4,4,4,4),dtype=tc.float64,device='cuda')

    for q in range(4):
        for w in range(4):
            for e in range(4):
                for r in range(4):
                    if q != w and w !=e and e!=r and q!=r:
                        if q == e or w == r:
                            if q ==e and w == r:
                                p[q,w,e,r] = (1-v)/2**8
                            else:
                                p[q,w,e,r] = (3+5*v)/((2**8)*3)

                            
                        else:
                            p[q,w,e,r] = (3+13*v)/((2**8)*3)
                    else:
                        p[q,w,e,r] = (1-v)/2**8

    return p
object_list = obj_list()



model = Qu4().to('cuda')

start_time = time.time()
laptime =[]

optimizer = tc.optim.Adam(model.parameters(), lr=learning_rate)  # Use Adam for optimization.

model.optimize(optimizer,epochs=iteration_number)

#model.draw()
    
#print('equivalent_v = ',(result_list[0]*8-1)/3)

laptime.append(time.time())

interval_list=[laptime[0]-start_time]
#for i in range(1):
#    interval_list.append(laptime[i+1]-laptime[i])
print('interval time:',interval_list)
#print('total loop time',laptime[11]-start_time)
#print('total time',laptime[11]-initial_time)







#p =  {'para': model.params, 'psi01': psi01, 'result':result_list, 'm1':m1}


#tc.save(p,'qu4_0.362.pt')





