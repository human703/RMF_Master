# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:42:27 2026

@author: varsh
"""


from scipy.optimize import root_scalar, fsolve, root
from scipy.integrate import quad
from numpy import sqrt, pi, log
import matplotlib.pyplot as plt

print("---------------------Exercsie 1---------------------------")
m=939#MeV
h_bar_c=197.327#MeV fm

def I1_k(k):
    return k**2/(sqrt( (h_bar_c*k)**2 +m**2))

def I1_p(p):
    return p**2/((h_bar_c**3)*(sqrt( p**2 +m**2)))
kf=1.36
result,error=quad(I1_k,0,kf)
result=float(result)
print("\nk integral evaluated value=", result)

pf=h_bar_c*kf
result,error=quad(I1_p,0,pf)
result=float(result)
print("p integral evaluated value=", result)

Ef=sqrt(pf**2+m**2)
# print("pf",pf, sqrt(Ef**2-m**2))
# print(kf,pf,Ef)
# print((0.5/(h_bar_c**3)))
# print(Ef*pf)
# print(Ef/pf)

I1_analytical= (1/(h_bar_c**3))*(0.5*Ef*pf + 0.25*(m**2)*log((Ef-pf)/(Ef+pf)))
print("Analytically evaluated value=", I1_analytical)

print("\n---------------------Exercsie 2---------------------------")

def phi(m):
    def integral(k):
        return (m*k**2)/(sqrt(k**2+m**2))
    result,error=quad(integral,0,1)
    result=float(result)
    return result

m_old=1.0
i=0
eps=10
M=[]
I=[]
M.append(m_old)
I.append(i)
while eps>1e-8:
    m_new=1-0.5*phi(m_old)
    eps=abs(m_new-m_old)
    m_old=m_new
    i=i+1
    M.append(m_old)
    I.append(i)

print("\nConverged value of m=",m_old)
print("number of iterations required: ",i)

plt.plot( I, M, 'b')
plt.ylabel('m_n')
plt.xlabel("number of iterations")
#plt.xlim(3,8)
#plt.ylim(0.8733,0.8734)
plt.show() 

print("\n---------------------Exercsie 3---------------------------")
g_sigma=300#Mev.fm^(3/2)

def residual(m_star):    
    Ef_star=sqrt(pf**2 + m_star**2)
    
    return m_star - m + ( (g_sigma**2)/((h_bar_c**3)*(pi**2)) )*( pf*Ef_star -(m_star**2)*log((pf+Ef_star)/(m_star)) )

result=root_scalar(residual, method='brentq',bracket=(0.001*m,m))
print("\nRoot calculated using brentq=" ,result.root)
print("Number of function calls: ",result.function_calls)

result=root_scalar(residual, method='secant',x0=0.00001*m, x1=m)
print("\nRoot calculated using secant=",result.root)
print("Number of function calls: ",result.function_calls)

print("\nOut of all the root finding methods, secant and brentq seem to be the best. Secant usually has fewer function calls and is hence faster. Both arrive at almost the same root but it is hard to tell which is closer to the real value. ")

print("\n---------------------Exercsie 4---------------------------")

m_star=800#MeV
V0=300#MeV
mu_target=1200#MeV

def g(pf):
    return sqrt(pf**2+m_star**2) + V0 - mu_target

result=fsolve(g,300)
print("\npf=",(result[0]),"MeV")
#print(type(result[0]))
print("kf=",(result[0])/h_bar_c, "fm^-1")

print("\n---------------------Exercsie 5---------------------------")

mu=2.5      
a_sigma=0.1
a_omega=0.05
g_sigma=0.5
g_omega=0.4

def F(x): 
    '''
    x must be a list containing guess values: [sigma,omega], 
    returns a list (sigma, omega)
    '''
    #print(type(x))
    x[0]=a_sigma*(mu- g_omega*x[1]-g_sigma*x[0])**2 #for sigma
    x[1]=a_omega*(mu- g_omega*x[1]-g_sigma*x[0])**3 #for omega
    
    x_new=[x[0],x[1]]
    return x_new

#print((F([1,2])))

Guess=[0.2,0.2]

print("\nUsing hybr")
result=root(F, x0=Guess,method='hybr')

if result.success==True:
    #print("True loop entered")
    print("sigma= ",result.x[0], "omega= ",result.x[1])
elif result.success==False:
    #print("False loop entered")
    print("Error in finding root. Give better guess!")

print("\nUsing krylov (seems wrong)")

result=root(F, x0=Guess,method='krylov')
#print(result)
if result.success==True:
    #print("True loop entered")
    print("sigma= ",result.x[0], "omega= ",result.x[1])
elif result.success==False:
    #print("False loop entered")
    print("Error in finding root. Give better guess!")
    
print("\nUsing lm")

result=root(F, x0=Guess,method='lm')
#print(result)
if result.success==True:
    #print("True loop entered")
    print("sigma= ",result.x[0], "omega= ",result.x[1])
elif result.success==False:
    #print("False loop entered")
    print("Error in finding root. Give better guess!")
    
print("\nOnly hybr, krylov and lm converge to a solution. hybr and lm converge to the same solution. krylov seems to give a very different solution.")