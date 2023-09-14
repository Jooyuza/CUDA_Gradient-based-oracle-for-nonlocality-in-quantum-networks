# -*- coding: utf-8 -*-


import torch as tc
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time

initial_time = time.time()
print(initial_time)

class Qu4(nn.Module):
    loss_history = [10]

    def __init__(self):
        super(Qu4, self).__init__()

        # Define the parameters of your model here.
        self.params = nn.Parameter(tc.randn(636,dtype=tc.float64),requires_grad=True)


        v = 0.363
        self.object_list = self.obj_list(v)

    def generate_state(self,a):
        a = tc.abs(a)
        
        a = nn.functional.normalize(a,dim=0,eps=1e-14)
        
        a = tc.unsqueeze(a,-1)
        



        psi01 = tc.cat((a[0],tc.zeros(6,device=self.params.device),a[1],tc.zeros(6,device=self.params.device),a[2], tc.zeros(6,device=self.params.device), a[3] , tc.zeros(6,device=self.params.device),a[4], tc.zeros(6,device=self.params.device),a[5]))
        psi23 = psi01
        psi45 = psi01

        psi0123 = tc.kron(psi01,psi23)
        psi012345 = tc.kron(psi0123,psi45)

        return psi012345
    
    

    def swap_cyclic(self,state):

        return tc.matmul(swap,state)

    def density_matrix(self,state):
        return tc.outer(state,state.conj())

    def povm(self,density,m1,m2,m3):
        ma = tc.kron(tc.kron(m1,m2),m3)
        mb = tc.matmul(density,ma)
        return tc.trace(mb)

    def generate_POVM(self,x):


        params = x
        A = tc.zeros((36, 36), dtype=tc.float64,device=self.params.device)
        D_i = tc.ones(r,dtype=tc.float64,device=self.params.device)

        Z = tc.zeros(36-r,dtype=tc.float64,device=self.params.device)

        B =tc.cat([D_i,Z])
        # parameterize the diagonal matrix D
        D = tc.diag((B))

        # parameterize the skew-symmetric matrix A
        idx = tc.triu_indices(36, 36, offset=1)  # indices for the upper triangular part excluding the diagonal
        A[idx[0], idx[1]] = params
        A = A - A.T  # to make A skew-symmetric

        # generate an orthogonal matrix O using the skew-symmetric matrix
        M = tc.eye(36,device=self.params.device) + A
        U = tc.mm(M, tc.inverse(tc.eye(36,device=self.params.device) - A))

        m1 = tc.mm(tc.mm(U, D), U.t())

        return m1

    def povm_result(self,rou,m1,m1_,m2,m2_,m3,m3_):
        povm_000 = self.povm(rou,m1,m2,m3)
        povm_001 = self.povm(rou,m1,m2,m3_)
        povm_010 = self.povm(rou,m1,m2_,m3)
        povm_011 = self.povm(rou,m1,m2_,m3_)
        povm_100 = self.povm(rou,m1_,m2,m3)
        povm_101 = self.povm(rou,m1_,m2,m3_)
        povm_110 = self.povm(rou,m1_,m2_,m3)
        povm_111 = self.povm(rou,m1_,m2_,m3_)

        result_list = [povm_000,povm_001,povm_010,povm_011,povm_100,povm_101,povm_110,povm_111]

        return result_list

    def obj_list(self,v):

        p = [0]*8
        for i in range (8):
            if i == 0:
                p[i] = (1+3*v)/8
            elif i == 7:
                p[i] = (1+3*v)/8
            else:
                p[i] = (1-v)/8
        return p

    def obj_fun(self,list_obj,list_m):
        kl= 0
        for i in range(8):
            kl += list_obj[i] * tc.log(list_obj[i]/list_m[i])
        return kl

    def loss_KL(self):

        state = self.generate_state(self.params[0:6])

        state_swaped = self.swap_cyclic(state)
        rou = self.density_matrix(state_swaped)

        m1 = self.generate_POVM(self.params[6:636])
        m1_ = tc.eye(36,device=self.params.device) - m1
        m2 = m1
        m2_ = tc.eye(36,device=self.params.device) - m2
        m3 = m1
        m3_ = tc.eye(36,device=self.params.device) - m3

        result_list = self.povm_result(rou,m1,m1_,m2,m2_,m3,m3_)


        y = self.obj_fun(self.object_list,result_list)

        return y

    def optimize(self,optimizer, epochs=10):

        for i in range(epochs):
            optimizer.zero_grad()  # clear previous gradients
            loss = self.loss_KL()  # compute the loss
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
            if i%1 == 0:
                print(i,loss)


    def draw(self):
        plt.figure(figsize=(10, 6))
        plt.plot(Qu4.loss_history)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over time')
        plt.savefig('loop_0.3621_pvm_r'+str(r)+'.png')

    def show_state(self,a):
        a = tc.abs(a)
        
        a = nn.functional.normalize(a,dim=0,eps=1e-14)
        
        a = tc.unsqueeze(a,-1)
        


        psi01 = tc.cat((a[0],tc.zeros(6,device=self.params.device),a[1],tc.zeros(6,device=self.params.device),a[2], tc.zeros(6,device=self.params.device), a[3] , tc.zeros(6,device=self.params.device),a[4], tc.zeros(6,device=self.params.device),a[5]))
 
        
        return psi01


def generate_swapmatrix():

    d = 6
    s = tc.zeros(d**2, d**2,dtype=tc.float64,device = 'cuda')

    for i in range(d):
        for j in range(d):
            row = i * d + j
            col = j * d + i
            s[row, col] = 1



    s01 = tc.kron(s,tc.eye(6**4,device = 'cuda'))

    s12_ = tc.kron(tc.eye(6,device = 'cuda'),s)
    s12 = tc.kron(s12_,tc.eye(6**3,device = 'cuda'))

    s23_ = tc.kron(tc.eye(6**2,device = 'cuda'),s)
    s23 = tc.kron(s23_,tc.eye(6**2,device = 'cuda'))

    s34_ = tc.kron(tc.eye(6**3,device = 'cuda'),s)
    s34 = tc.kron(s34_,tc.eye(6,device = 'cuda'))

    s45 = tc.kron(tc.eye(6**4,device = 'cuda'),s)

    swap_cyclic = tc.einsum('ab,bc,cd,de,ef->af', s45, s34, s23, s12, s01)

    return swap_cyclic
swap = generate_swapmatrix()
#swap=swap.to('cuda')

model = Qu4().to('cuda')

start_time = time.time()
laptime =[]
for i in range(1):
    r=18
    print('r=:',r)
    optimizer = tc.optim.Adam(model.parameters(), lr=0.01)  # Use Adam for optimization.

    model.optimize(optimizer)

    model.draw()
    psi01 = model.show_state(model.params[0:6])
    print(psi01)

    #print(model.params)

    state = model.generate_state(model.params[0:6])

    state_swaped = model.swap_cyclic(state)
    rou = model.density_matrix(state_swaped)

    m1 = model.generate_POVM(model.params[6:636])
    m1_ = tc.eye(36,device=model.params.device) - m1
    m2 = m1
    m2_ = tc.eye(36,device=model.params.device) - m2
    m3 = m1
    m3_ = tc.eye(36,device=model.params.device) - m3

    result_list = model.povm_result(rou,m1,m1_,m2,m2_,m3,m3_)
    print('equivalent_v = ',(result_list[0]*8-1)/3)
    print(result_list)
    laptime.append(time.time())

interval_list=[laptime[0]-start_time]
#for i in range(1):
#    interval_list.append(laptime[i+1]-laptime[i])
print('interval time:',interval_list)
#print('total loop time',laptime[11]-start_time)
#print('total time',laptime[11]-initial_time)







#p =  {'para': model.params, 'psi01': psi01, 'result':result_list, 'm1':m1}


#tc.save(p,'qu4_0.362.pt')





