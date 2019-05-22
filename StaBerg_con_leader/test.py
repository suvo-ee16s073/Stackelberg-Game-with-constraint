import numpy as np
from C_on_leader.Main import Main


######### matrics and dimensions
# K = number of stages (0 -> T)
# n = x_k dimension of states
# m_1 = u_k (follower) dimension    f
# m_2 = w_k (leader) dimension
# c = v_k leader's control from constraint
# s = constrainghts dim


K = 30
n = 1
m_1 = 1
m_2 = 1
c = 2
s = 2

#K = 3
#n = 2
#m_1 = 2
#m_2 = 2
#c = 2
#s = 2


A = np.matrix([[10]])
B_1 = np.matrix([[2]])
B_2 = np.matrix([[1]])
Q_1  = np.matrix([[2]])
Q_2 = np.matrix([[4]])
R_11 = np.matrix([[1]])
R_22 = np.matrix([[7]])
R_12 = np.matrix([[1]])
R_21 = np.matrix([[1]])
Q_1b = np.matrix([[1]])
Q_2b = np.matrix([[2]])
M = np.matrix([[2],[1]])
N = np.matrix([[1, 3],[3, 5]])
r = np.matrix([[0], [8]])
D = np.matrix([[1, 0], [0, 24]])
L = np.matrix([[1, 8]])
L_Bar = np.matrix([[9, 6]])
Sf_bar = np.matrix([[6]])
Sf_bar_b = np.matrix([[9]])
Sl_bar = np.matrix([[3]])
Sl_bar_b = np.matrix([[5]])
x0 = np.matrix([[-70]])

L_Bar = np.matrix([[2, 5]])
Sf_bar = np.matrix([[3]])
Sf_bar_b = np.matrix([[0]])
Sl_bar = np.matrix([[50]])
Sl_bar_b = np.matrix([[1]])

#A = np.matrix([[1, 1],[3, 7]])
#B_1 = np.matrix([[1], [1]])
#B_2 = np.matrix([[1], [4]])
#Q_1  = np.matrix([[2, 0],[0, 4]])
#Q_2 = np.matrix([[4, 0], [0, 1]])
#R_11 = np.matrix([[1]])
#R_22 = np.matrix([[1]])
#R_12 = np.matrix([[2]])
#R_21 = np.matrix([[1]])
#Q_1b = np.matrix([[1, 0], [0, 1]])
#Q_2b = np.matrix([[1, 0], [0, 1]])
#M = np.matrix([[2, 1],[1, 10]])
#N = np.matrix([[1, 2],[3, 7]])
#r = np.matrix([[0], [8]])
#D = np.matrix([[14, 0], [7, 0]]) 
#L = np.matrix([[1, 8],[4, 5]])
#x0 = np.matrix([[1],[1]])

#obj = Main( K, n, m_1, m_2, c, s, A, B_1, B_2, M, N, r, Q_1, R_11, R_12, Q_1b, Q_2, R_21, R_22, D, L, Q_2b, x0)
obj = Main(K, n, m_1, m_2, c, s, A, B_1, B_2, M, N, r, Q_1, R_11, R_12, Q_1b, Q_2, R_21, R_22, D, L, L_Bar, Q_2b, Sf_bar, Sf_bar_b, Sl_bar, Sl_bar_b, x0)
follower_uc = obj.follower_noncoupled()
leader_uc = obj.leader_noncoupled()
matrix_1 = obj.Build_matrics_1()
matrix_2 = obj.Build_matrics_2()
#Delta_p = obj.delta_p()
Delta_p = obj.delta_p_() 
Delta_0 = obj.delta_0()  
P = obj.LCP(Delta_p, Delta_0)
Xi = obj.Xi(Delta_0, Delta_p, P[0])
zeta = obj.Zeta(matrix_1['Ck_bar'], matrix_1['Fk_bar'], P, Xi )
Zeta_ = obj.Zeta_(matrix_1['Ck_bar'], matrix_1['Fk_bar'], P, Xi, matrix_1['phi_k'])
wk = obj.wk(leader_uc['gamma2'], matrix_1['Fk'],matrix_1['Ek'], zeta[1], zeta[2])
xk, uk_,zeta1 = obj.uk_(follower_uc['gamma1'], follower_uc['P1'], zeta[1], wk, zeta[2])




#######################################################################################
'''                           test                                                  '''
#######################################################################################
""" TEST 1"""
upsilon = follower_uc['upsilon1']
gamma1 = follower_uc['gamma1']
temp1_ = x0
temp11_ = x0
for ii in range(1, obj.K+2):
    temp1_ = np.dot( np.dot(upsilon[ii],A), temp1_) +  np.dot( np.dot(upsilon[ii], B_2),  wk[0,ii-1]) - np.dot(np.dot(B_1, np.linalg.inv(gamma1[ii])),np.dot(B_1.T,zeta1[0,ii]))
    temp11_ = np.concatenate((temp11_, temp1_), axis = 1)
    
    
""" TEST 2"""  
Ck = matrix_1['Ck']
Dk = matrix_1['Dk']
Ek = matrix_1['Ek']
temp2 = obj.zeta_0
temp22 = temp2
for ii in range(1, obj.K+2):
    temp2 =  np.dot(Ck[ii-1], temp2 )  +   np.dot(Dk[ii-1], zeta[2][:,ii] ) +np.dot(Ek[ii-1], wk[0,ii-1] )
    temp22 = np.concatenate((temp22, temp2), axis = 1)
    
    
""" TEST 3"""
temp3 = x0
temp33 = temp3
for ii in range(0, obj.K+1):
    temp3 =   np.dot(A, temp3)+np.dot(B_2,wk[:,ii])+np.dot(B_1,uk_[:,ii])
    temp33 = np.concatenate((temp33, temp3), axis = 1)
    
""" TEST 4"""
P1  = follower_uc['P1']  
temp4_ = x0
temp44_ = x0    
for ii in range(1, obj.K+2):
    temppp =  np.dot(A, temp4_) +  np.dot(B_2,  wk[0,ii-1])
    temp4_ = temppp - np.dot(np.dot(B_1, np.linalg.inv(gamma1[ii])),np.dot(B_1.T,(np.dot(P1[ii], temppp)+zeta1[0,ii])))
    temp44_ = np.concatenate((temp44_, temp4_), axis = 1)