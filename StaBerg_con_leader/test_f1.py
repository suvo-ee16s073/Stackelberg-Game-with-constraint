import numpy as np
from C_on_leader.Main import Main

######### matrics and dimensions
# K = number of stages (0 -> T)
# n = x_k dimension of states
# m_1 = u_k (follower) dimension    f
# m_2 = w_k (leader) dimension
# c = v_k leader's control from constraint
# s = constrainghts dim


K = 10#
n = 2#
m_1 = 1#
m_2 = 1#
c = 1#
s = 1#




A = np.matrix([ [ 0.1, 0], [ 0, 0.1]])#
B_2 = np.matrix([[1], [0]])# leader
B_1 = np.matrix([[0],[1]])# follower
Q_1  = 2*np.dot(np.matrix([[0], [1]]), np.matrix([[0, 0.5]])) #follower
Q_2 = 2*np.matrix([[0, 0], [0, 0]])# leader 
R_11 = np.matrix([[0.75]]) # follower
R_22 = np.matrix([[0.75]]) #leader
R_12 = np.matrix([[0]])# 
R_21 = np.matrix([[0]])#
Q_1b = 2*np.matrix([[0, 0], [0, -0.25]]) + 2*Q_1 # follower
Q_2b = 2*np.matrix([[-0.25, 0], [0, 0]]) # leader
M = np.matrix([[1, 0]])#
N = np.matrix([[-1]])#
r = np.matrix([[0]])#
#M = np.matrix([[0, 0, 1, 0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
#N = np.matrix([[-1],[0],[0],[0],[0]])
#r = np.matrix([[0],[0],[0],[0],[0]])

D = 2*np.matrix([[0.5]])# leader
L =  np.matrix([[0], [0.5]])#

L_Bar = np.matrix([[0], [0.5]]) #
Pl_bar = np.matrix([[0.5]]) - np.matrix([[3]]) #
Sf_bar = np.matrix([[0], [0.5]]) - np.matrix([[0], [3]]) #
Sf_bar_b = Sf_bar#
Sl_bar = np.matrix([[0], [0]])#
Sl_bar_b = np.matrix([[0], [0]])#

x0 = np.matrix([[9], [2]])#

#
obj = Main(K, n, m_1, m_2, c, s, A, B_1, B_2, M, N, r, Q_1, R_11, R_12, Q_1b, Q_2, R_21, R_22, D, L, L_Bar, Q_2b, Sf_bar, Sf_bar_b, Sl_bar, Sl_bar_b, Pl_bar, x0)
follower_uc = obj.follower_noncoupled()
leader_uc = obj.leader_noncoupled()
matrix_1 = obj.Build_matrics_1()
matrix_2 = obj.Build_matrics_2()
#Delta_p = obj.delta_p()
Delta_p = obj.delta_p_() 
Delta_0 = obj.delta_0()  
P = obj.LCP(Delta_p, Delta_0)
if P[0] is None:
    print("LCP not solvable")
else:
    Xi = obj.Xi(Delta_0, Delta_p, P[6], P[0])
    #zeta = obj.Zeta(matrix_1['Ck_bar'], matrix_1['Fk_bar'], P, Xi )
    Zeta_ = obj.Zeta_(matrix_1['Ck_bar'], matrix_1['Fk_bar'], P, Xi, matrix_1['phi_k'])
    wk = obj.wk(leader_uc['gamma2'], matrix_1['Fk'],matrix_1['Ek'], Zeta_[1], Zeta_[2])
    xk, uk_,zeta1 = obj.uk_(follower_uc['gamma1'], follower_uc['P1'], Zeta_[1], wk, Zeta_[2])




    #######################################################################################
    '''                           test                                                  '''
    #######################################################################################
    
    """ TEST 3"""
    temp3 = x0
    temp33 = temp3
    for ii in range(0, obj.K+1):
        temp3 =   np.dot(A, temp3)+np.dot(B_2,wk[:,ii])+np.dot(B_1,uk_[:,ii])
        temp33 = np.concatenate((temp33, temp3), axis = 1)
    
