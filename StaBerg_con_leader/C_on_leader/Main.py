import numpy as np
from scipy import linalg
from lemkelcp.lemkelcp import lemkelcp as lcp

######### matrics and dimensions
# K = number of stages (0 -> T)
# n = x_k dimension of states
# m_1 = u_k (follower) dimension    f
# m_2 = w_k (leader) dimension
# c = v_k leader's control from constraint
# s = constrainghts dim

class Main:
    def __init__(self, K, n, m_1, m_2, c, s, A, B_1, B_2, M, N, r, Q_1, R_11, R_12, Q_1b, Q_2, R_21, R_22, D, L, L_Bar, Q_2b, Sf_bar, Sf_bar_b, Sl_bar, Sl_bar_b, Pl_bar, x0):
        ## dimensions
        self.K = K                 
        self.n = n
        self.m_1 = m_1
        self.m_2 = m_2
        self.c = c
        self.s = s
        ## System equation 
        self.A = A
        self.B_1 = B_1
        self.B_2 = B_2
        ## constrainst
        self.M = M
        self.N = N
        self.r = r
        ## Follower's objective
        self.Q_1 = Q_1
        self.R_11 = R_11
        self.R_12 = R_12
        self.Q_1b = Q_1b
        ## Leader's objective
        self.Q_2 = Q_2
        self.R_21 = R_21
        self.R_22 = R_22
        self.D = D
        self.L = L 
        self.L_Bar = L_Bar
        self.Q_2b = Q_2b
        self.x0 = x0
        self.Sl_bar = Sl_bar
        self.Sl_bar_b = Sl_bar_b
        self.Sf_bar = Sf_bar
        self.Sf_bar_b = Sf_bar_b
        self.Pl_bar = Pl_bar
        ## initialization before computation  
        self.gamma1 = {}
        self.upsilon1 = {}
        self.P1 = {}
        
        self.gamma2 = {}
        self.S21 = {}
        self.P2 = {} 
        
        #self.G = np.concatenate((np.concatenate((np.zeros(shape = (self.n,self.c)), np.zeros(shape = (self.n, self.s))), axis = 1),np.concatenate((self.L, -self.M.T), axis = 1)))
        self.G = np.concatenate((np.concatenate((self.L_Bar, np.zeros(shape = (self.n, self.s))), axis = 1),np.concatenate((self.L, -self.M.T), axis = 1)))
        self.Sbar_b =  np.concatenate((self.Sf_bar_b, self.Sl_bar_b), axis = 0)
        self.Sbar =  np.concatenate((self.Sf_bar, self.Sl_bar), axis = 0)
        self.S = np.concatenate((np.kron(np.ones((self.K, 1)), self.Sbar), self.Sbar_b), axis = 0)
        self.Ck = {}
        self.Dk = {}
        self.Ek = {}
        self.Fk = {}
        self.Ck_bar = {}
        self.Dk_bar = {}
        self.Fk_bar = {}
        self.phi_k = {}
        self.Ak_bar = {}
        self.Bk_bar = {}
        self.xi_k = {}
        ## lcp 
        self.M_ = np.concatenate((np.concatenate((self.D, -self.N.T), axis = 1),np.concatenate((self.N, np.zeros((self.c, self.c))), axis = 1)))
        self.aa = np.concatenate((np.zeros(shape = (self.s,self.n)), self.L.T), axis = 1)
        self.bb = np.concatenate((np.zeros(shape = (self.c, self.n)), self.M), axis = 1)
        self.q = np.concatenate((self.aa, self.bb) )
        self.zeta_0 = np.concatenate((np.zeros((self.n, 1)), self.x0))

        
        
    def follower_noncoupled(self):
        self.P1.update({self.K+1 : self.Q_1b})
        self.gamma1.update({self.K+1 : self.R_11 + np.dot(np.dot(self.B_1.T,self.P1[self.K+1]),self.B_1) })  
        self.upsilon1.update({self.K+1 : np.eye(self.n) - np.dot(np.dot(np.dot(self.B_1 ,  linalg.inv(self.gamma1[self.K+1])), self.B_1.T) , self.P1[self.K+1]) })
        for ii in range(self.K, -1, -1):  # K to 0, middle -1 count to 0 or it stop at 1
            self.P1.update({ii : self.Q_1 + np.dot(np.dot(np.dot(self.A.T, self.P1[ii+1]), self.upsilon1[ii+1]),  self.A)})
            self.gamma1.update({ii : self.R_11 + np.dot(np.dot(self.B_1.T,self.P1[ii]),self.B_1) })  
            self.upsilon1.update({ii :  np.eye(self.n) - np.dot(np.dot(np.dot(self.B_1 ,  linalg.inv(self.gamma1[ii])), self.B_1.T) , self.P1[ii]) })
        return {'P1' : self.P1, 'gamma1' : self.gamma1, 'upsilon1' : self.upsilon1}


    def leader_noncoupled(self):
        if(len(self.P1) == 0 & len(self.gamma1) == 0 & len(self.upsilon1) == 0):
            print("Compute for follower's non-coupled parameters first before doing leader")
        else:
            self.S21.update({self.K+1 : linalg.inv(self.gamma1[self.K+1]) * self.R_21 * linalg.inv(self.gamma1[self.K+1])})
            self.P2.update({self.K+1 : self.Q_2b})
            self.gamma2.update({self.K+1 : self.R_22 + self.B_2.T * self.P1[self.K+1] * self.B_1 * self.S21[self.K+1] * self.B_1.T * self.P1[self.K+1] * self.B_2  + self.B_2.T * self.upsilon1[self.K+1].T * self.P2[self.K+1] * self.upsilon1[self.K+1] * self.B_2 }) 
            for ii in range(self.K, -1, -1):
                self.S21.update({ii : linalg.inv(self.gamma1[ii]) * self.R_21 * linalg.inv(self.gamma1[ii])})
                self.P2.update({ii :  self.Q_2 + self.A.T * self.upsilon1[ii+1].T * self.P2[ii+1] * self.upsilon1[ii+1] * self.A + self.A.T * self.P1[ii+1] * self.B_1 * self.S21[ii+1] * self.B_1.T * self.P1[ii+1]* self.A})
                self.gamma2.update({ii : self.R_22 + self.B_2.T * self.P1[ii] * self.B_1 * self.S21[ii] * self.B_1.T * self.P1[ii] * self.B_2  + self.B_2.T * self.upsilon1[ii].T * self.P2[ii] * self.upsilon1[ii] * self.B_2 }) 
        return {'P2' : self.P2, 'gamma2' : self.gamma2, 'S21' : self.S21}
    
    def Build_matrics_1(self):
        if(len(self.P2) == 0 & len(self.gamma2) == 0 & len(self.S21) == 0):
            print("Compute non-coupled parameters first")
        else:
            self.phi_k[self.K+1] = np.zeros(shape = (2*self.n, 2*self.n))
            for ii in range(self.K, -1, -1):
                ac = self.upsilon1[ii+1] * self.A
                bc = self.B_1 * self.S21[ii+1] * self.B_1.T * self.P1[ii+1] * self.A - self.B_1 * linalg.inv(self.gamma1[ii+1]) * self.B_1.T * self.P2[ii+1] * self.upsilon1[ii+1] * self.A
                cc = np.zeros(shape=(self.n, self.n))
                self.Ck[ii] = np.concatenate((np.concatenate((ac,bc), axis = 1),np.concatenate((cc, ac), axis = 1)))
                ad = self.B_1 * (self.S21[ii+1] + linalg.inv(self.gamma1[ii+1]) * self.B_1.T * self.P2[ii+1] * self.B_1 * linalg.inv(self.gamma1[ii+1]) )* self.B_1.T
                bd = -self.B_1 * linalg.inv(self.gamma1[ii+1]) * self.B_1.T 
                self.Dk[ii] = np.concatenate((np.concatenate((ad,bd), axis = 1),np.concatenate((bd, cc), axis = 1)))
                ae = self.B_1 * (self.S21[ii+1] * self.B_1.T * self.P1[ii+1] - linalg.inv(self.gamma1[ii+1]) * self.B_1.T * self.P2[ii+1] * self.upsilon1[ii+1]) * self.B_2
                ce = self.upsilon1[ii+1] * self.B_2
                self.Ek[ii] = np.concatenate((ae,ce))
                af = self.A.T * self.upsilon1[ii+1].T * self.P1[ii+1] * self.B_2
                cf = self.A.T * (self.upsilon1[ii+1].T* self.P2[ii+1] * self.upsilon1[ii+1] + self.P1[ii+1] * self.B_1 * self.S21[ii+1] * self.B_1.T * self.P1[ii+1]) * self.B_2
                self.Fk[ii] = np.concatenate((af,cf))
                self.Ck_bar[ii] = self.Ck[ii] - self.Ek[ii] * linalg.inv(self.gamma2[ii+1]) * self.Fk[ii].T
                self.Dk_bar[ii] = self.Dk[ii] - self.Ek[ii] * linalg.inv(self.gamma2[ii+1]) * self.Ek[ii].T
                self.Fk_bar[ii] = - self.Fk[ii] * linalg.inv(self.gamma2[ii+1]) * self.Fk[ii].T
                self.phi_k[ii] = self.Ck_bar[ii].T * self.phi_k[ii+1] * linalg.inv(np.eye(2*self.n)-self.Dk_bar[ii] * self.phi_k[ii+1]) * self.Ck_bar[ii] + self.Fk_bar[ii]
        return {'Ck' : self.Ck, 'Dk' : self.Dk, 'Ek' : self.Ek, 'Fk' : self.Fk, 'Ck_bar' : self.Ck_bar, 'Dk_bar' : self.Dk_bar, 'Fk_bar' : self.Fk_bar, 'phi_k' : self.phi_k}

    def Build_matrics_2(self):
        if(len(self.phi_k) == 0):
            print(" Compute phi_k first")
        else:
            for ii in range(self.K, -1, -1):
                temp1 = linalg.inv(np.eye(2*self.n) - self.Dk_bar[ii] * self.phi_k[ii+1])
                self.Ak_bar.update({ii : temp1 * self.Ck_bar[ii]})
                self.Bk_bar.update({ii : temp1 * self.Dk_bar[ii]})
                self.xi_k.update({ii : self.Ck_bar[ii].T * linalg.inv(np.eye(2*self.n) - self.phi_k[ii+1] * self.Dk_bar[ii])}) 
        return {'Ak_bar' : self.Ak_bar, 'Bk_bar' : self.Bk_bar, 'xi_k' : self.xi_k}
    def epsilon_m_l(self, m, l):
        if(len(self.Ak_bar) == 0):
            print(" Compute Ak_bar first")
        else:
            epsilon = np.eye(2*self.n)
            if(l < m):
                for ii in range(m-1, l-1, -1):
                    epsilon = epsilon * self.Ak_bar[ii]
            else:
                if(l > m):
                    epsilon = np.zeros(shape = (2*self.n, 2*self.n))
        return epsilon
        
    def Delta_ml(self, m, l):    
        if(len(self.xi_k) == 0):
            print(" Compute xi_k first")
        else:
            if(l < m):
                Delta = np.eye(2*self.n)
                for ii in range(m-1, l-1, -1):
                    Delta = self.xi_k[ii] * Delta
                Delta = Delta * self.G
            else:
                if(l == m):
                    Delta  = self.G  
                else:
                    Delta = np.zeros(shape = (2*self.n, (self.s + self.c)))
        return Delta  

    def Delta_ml_S(self, m, l):    
        if(len(self.xi_k) == 0):
            print(" Compute xi_k first")
        else:
            if(l < m):
                Delta = np.eye(2*self.n)
                for ii in range(m-1, l-1, -1):
                    Delta = self.xi_k[ii] * Delta
            else:
                if(l == m):
                    Delta  = np.eye(2*self.n)
                else:
                    Delta = np.zeros(shape = (2*self.n, 2*self.n))
        return Delta       
        
    def deltap_kj(self, k, j):     
        if(len(self.Bk_bar) == 0):
            print(" Compute Bk_bar first")
        else:    
            minimum = np.min(a = [k, j])
            deltap_kj = np.zeros(shape = (2*self.n, (self.s + self.c)))
            for ii in range(1, minimum + 1):
                temp = np.dot((self.epsilon_m_l(k, ii) * self.Bk_bar[ii-1]),  self.Delta_ml(j, ii))
                deltap_kj = deltap_kj + temp 
        return deltap_kj

    def delta_p_(self):
        temp_axis_1 = self.deltap_kj(1, 1)
        for col in range(2, self.K+2):
            temp_axis_1 = np.concatenate((temp_axis_1, self.deltap_kj(1, col)), axis = 1)
            
        for row in range(2, self.K+2):
            for col in range(1, self.K+2):
                if(col == 1):
                    temp = self.deltap_kj(row, col)
                else:
                    temp = np.concatenate((temp, self.deltap_kj(row, col)), axis = 1)
            temp_axis_1 = np.concatenate((temp_axis_1, temp), axis = 0)
        return temp_axis_1      

    def deltap_kj_S(self, k, j):     
        if(len(self.Bk_bar) == 0):
            print(" Compute Bk_bar first")
        else:    
            minimum = np.min(a = [k, j])
            deltap_kj = np.zeros(shape = (2*self.n, 2*self.n))
            for ii in range(1, minimum + 1):
                temp = np.dot((self.epsilon_m_l(k, ii) * self.Bk_bar[ii-1]),  self.Delta_ml_S(j, ii))
                deltap_kj = deltap_kj + temp 
        return deltap_kj

    def delta_p_S(self):
        temp_axis_1 = self.deltap_kj_S(1, 1)
        for col in range(2, self.K+2):
            temp_axis_1 = np.concatenate((temp_axis_1, self.deltap_kj_S(1, col)), axis = 1)
            
        for row in range(2, self.K+2):
            for col in range(1, self.K+2):
                if(col == 1):
                    temp = self.deltap_kj_S(row, col)
                else:
                    temp = np.concatenate((temp, self.deltap_kj_S(row, col)), axis = 1)
            temp_axis_1 = np.concatenate((temp_axis_1, temp), axis = 0)
        return np.dot(temp_axis_1, self.S)          
        
    def delta_0(self):
        temp = self.epsilon_m_l(1, 0)
        for row in range(2, self.K+2):
            temp = np.concatenate((temp, self.epsilon_m_l(row, 0)), axis = 0)
        return temp
      
    def LCP(self, delta_p, delta_0):
        delta_P_s = self.delta_p_S()
        I = np.eye(self.K+1)
        M_tilde = np.kron(I, self.M_)
        q_tilde = np.kron(I, self.q)
        s_tilde = np.kron(np.ones((self.K+1, 1)), np.concatenate( ( self.Pl_bar, self.r) ))
        Big_M = M_tilde + np.dot(q_tilde, delta_p)
        Big_q = s_tilde + np.dot(q_tilde, np.dot(delta_0, self.zeta_0)) + np.dot(q_tilde, delta_P_s)
        eigen_vals = np.linalg.eigvals(np.array([Big_M]))
        if eigen_vals.dtype == 'complex128':
            print("error : LCP matrix has complex eigen values")
            p = -1
            #return None
            return None, None, None, None, Big_M, Big_q, eigen_vals
        else:
            p = lcp(Big_M, Big_q)
            p_0 = lcp(self.M_, np.dot(self.q, self.zeta_0) + np.concatenate( ( np.zeros( (self.s, 1) ), self.r)))
            if p[0] is None:        
                p_ = None       
            else:
                p_ = np.reshape(p[0], (self.s+self.c, self.K+1),'F') 
            return np.array([p[0]]).T, np.array([p_0[0]]).T, p_, p, Big_M, Big_q, delta_P_s
        
        
    def Xi(self, delta_0, delta_p, delta_P_s, p):
        Xi = np.dot(delta_0, self.zeta_0) + np.dot(delta_p, p) + delta_P_s
        Xi_ = np.reshape(Xi, (2*self.n, self.K+1),'F')
        return Xi, Xi_
        ############################################### have to modify after this
    def Zeta(self, C_k_bar, F_k_bar , P, Xi): # modify xi and p
        p_k =  np.concatenate((P[1] ,P[2]), axis = 1)
        Xi =  np.concatenate((self.zeta_0 ,Xi[1]), axis = 1)
        zeta_K_plus_1 = np.dot(self.G, np.array([p_k[:, self.K]]).T)
        zeta =  zeta_K_plus_1
        temp = zeta 
        for ii in range(self.K, -1, -1):
            temp =  np.dot(C_k_bar[ii].T, temp) + np.dot(F_k_bar[ii], Xi[:, ii]) + np.dot(self.G, np.array([p_k[:, ii]]).T)
            zeta = np.concatenate((temp , zeta), axis = 1)
        return p_k, Xi, zeta
                
    def Zeta_(self, C_k_bar, F_k_bar , P, Xi, phi): # modify xi and p
        p_k =  np.concatenate((P[1] ,P[2]), axis = 1)
        Xi =  np.concatenate((self.zeta_0 ,Xi[1]), axis = 1)
        delta = {}
        for ii in range(0, self.K+2):
            temp = np.zeros(shape= (2*self.n, 1))
            for jj in range(ii, self.K+2):
                if(jj == self.K+1):
                    ss = self.Sbar_b
                else:
                    ss = self.Sbar
                temp = temp + np.dot(self.Delta_ml(jj, ii), np.array([p_k[:, jj]]).T) + np.dot(self.Delta_ml_S(jj, ii), ss) #add s terms eqs after 29 
                delta.update({ii : temp})
        zeta_K_plus_1 = np.dot(self.G, np.array([p_k[:, self.K]]).T) + self.Sbar_b #add s terms eq 26(b)
        zeta =  zeta_K_plus_1
        temp = zeta
        for ii in range(self.K, -1, -1):
            temp =  np.dot(phi[ii], Xi[:,ii]) + delta[ii]  #add s terms eq assumption 4
            zeta = np.concatenate((temp , zeta), axis = 1)        
        return p_k, Xi, zeta, delta
        
    def wk(self, gamma2, Fk, Ek, Xi, zeta):
        temp = (np.dot(Fk[self.K].T, Xi[:, self.K])+ np.dot(Ek[self.K].T, zeta[:, self.K+1]))
        wk = - np.dot(np.linalg.inv(gamma2[self.K+1]), temp)
        for ii in range(self.K-1, -1, -1):
             temp1 = - np.dot(np.linalg.inv(gamma2[ii+1]), (np.dot(Fk[ii].T, Xi[:, ii])+ np.dot(Ek[ii].T, zeta[:, ii+1])))   
             wk = np.concatenate((temp1, wk), axis = 1)
        return wk
        
    def uk_(self, gamma1, P1, Xi, wk , zeta):
        xk = Xi[list(range(self.n, self.n*2)), :]
        zeta1 = zeta[list(range(0, self.n)) , :]
        uk = {}
        temp4_ = self.x0  
        for ii in range(1, self.K+2):
            temppp =  np.dot(self.A, temp4_) +  np.dot(self.B_2,  wk[:,ii-1])
            temp4_ = temppp - np.dot(np.dot(self.B_1, np.linalg.inv(gamma1[ii])),np.dot(self.B_1.T,(np.dot(P1[ii], temppp)+zeta1[:,ii])))
            temp4 = -np.dot(np.dot(np.linalg.inv(gamma1[ii]), self.B_1.T),(np.dot(P1[ii], temppp)+zeta1[:,ii]))
            uk.update({ii-1 : temp4})
        temp1 = uk[0]
        for jj in range(1, self.K+1):
            temp1 = np.concatenate((temp1, uk[jj]), axis = 1)
        return xk, temp1, zeta1 