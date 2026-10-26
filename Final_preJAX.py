"""
Sigma-omega-rho model with non-linear terms in sigma+Beta equilibrium (only electrons)
Only with the full model. For comparison of B/A vs n for other models, see older programs
"""
from scipy.optimize import root
from scipy.integrate import quad
from numpy import sqrt, pi, linspace, zeros, savetxt
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

import Constants as cst

#Walecka parametrization, pg147 in pdf, Theoretical Nuclear and Subnuclear
# gs_ms_Walecka=sqrt(267.1/(cst.m_nucleon**2))
# gw_mw_Walecka=sqrt(195.9/(cst.m_nucleon**2))
# ms_gs_Walecka=1/gs_ms_Walecka
# mw_gw_Walecka=1/gw_mw_Walecka

#CONVENTION FOR PARAMETRIZAATION
'''
gs_ms and gw_mw MUST BE in units of MeV^-1. Units are taken care of in the gss function
'''
#Glendenning parametrization
gs_ms=sqrt(12.684)/cst.hbar_c
gw_mw=sqrt(7.148)/cst.hbar_c
gr_mr=sqrt(4.410)/cst.hbar_c
#Initialising fixed parameters
b=0.00561 #dimensionless
#b=0
c=-0.006986 #dimensionless
#c=0

#NL3 parametrization
# m_sigma = 508.194   # MeV
# m_omega = 782.501   # MeV
# m_rho   = 763.000   # MeV
# g_sigma = 10.217    # dimensionless
# g_omega = 12.868    # dimensionless
# g_rho   = 4.474     # dimensionless
# g2      = -10.431   # fm^-1
# g3      = -28.885   # dimensionless (MeV^-2 in natural units -- no hbar_c needed)

# # Convert g2: fm^-1 -> MeV^-1 * MeV = dimensionless b
# b = -(g2 * hbar_c) / (cst.m_nucleon * g_sigma**3)   # dimensionless
# c = g3 / (g_sigma**4)                              # dimensionless

# # Coupling ratios in MeV^-1
# gs_ms = g_sigma / m_sigma
# gw_mw = g_omega / m_omega
# gr_mr = g_rho   / m_rho

#GM1 parametrization

m_sigma = 512.0   # MeV
m_omega = 783.0   # MeV
m_rho   = 770.0  # MeV
g_sigma = 8.910    # dimensionless
g_omega = 10.610    # dimensionless
g_rho   = 4.093
b= 0.002947
c=-0.001070

#-------------------------------------------------------
gs_ms=g_sigma/m_sigma
gw_mw=g_omega/m_omega
gr_mr=g_rho/m_rho
ms_gs=1/gs_ms
mw_gw=1/gw_mw
mr_gr=1/gr_mr


#Defnining functions
# carefull with momenta
#kf different from pf=hbar*kf
def k_fermi(n): #eq 4.149, Glendenning  
    n_safe = max(n, 0.0) # Prevents negative values from crashing cube roots
    return (3*(pi**2)*n_safe/2)**(1/3)

def n_fermi(kf):
    return (2*kf**3)/(3*pi**2)

def k_fermi_i(n_i):#n_i is the number density of either neutrons or protons (vector density)
    n_safe_i = max(n_i, 0.0) # Prevents negative values from crashing cube roots
    return (3*(pi**2)*n_safe_i)**(1/3)

def n_fermi_i(kf_i):
    return (kf_i**3)/(3*pi**2)


def gss_EOM_numerical_self_int(num_density):   #Outputs the value of g_sigma*sigma at a given value of number density
    def gss(gss):
        kf=k_fermi(num_density)
        m_star=cst.m_nucleon-gss
        def integral_eval(kf):
            def integral(k):
                return (k**2)*(m_star)/(sqrt((cst.hbar_c*k)**2 + (m_star)**2))
            result,error=quad(integral,0,kf)
            result=float(result)
            return result
        return gss-(gs_ms**2)*( (2*(cst.hbar_c**3)/pi**2)*integral_eval(kf) -b*cst.m_nucleon*(gss**2) -c*(gss**3) )
    
    x0=[cst.m_nucleon/2] #guess for g_sigma*sigma taken as an intermediate value between 0 to m since it always lies between the two
    solution=root(gss,x0)
    sol=float(solution.x[0])
    return sol


def gww_EOM_self_int(num_density):   #Outputs the value of g_omega*omega_0 at a given value of number density
    return (cst.hbar_c**3)*(gw_mw**2)*num_density

def grr_EOM(num_density, Y_p):
    return (cst.hbar_c**3)*(gr_mr**2)*0.5*num_density*(2*Y_p-1)    


def energy_density_rho(num_density, Y_p):       #only nucleonic contribution
    n_p=num_density*Y_p
    n_n=num_density*(1-Y_p)
    
    gss=gss_EOM_numerical_self_int(num_density)
    gww=gww_EOM_self_int(num_density)
    grr=grr_EOM(num_density, Y_p)
    kf_p=k_fermi_i(n_p)
    kf_n=k_fermi_i(n_n)

    def integral_eval(kf):
        def integral(k):
            return (k**2)*(sqrt((cst.hbar_c*k)**2 + (cst.m_nucleon-gss)**2))
        result,error=quad(integral,0.,kf)
        result=float(result)
        return result
    
    E=( (ms_gs**2)*(gss**2) + (mw_gw**2)*(gww**2) + (mr_gr**2)*(grr**2))/(2*(cst.hbar_c**3)) + (1/cst.hbar_c**3)*( (1/3)*b*cst.m_nucleon*gss**3 + (1/4)*c*gss**4 ) + (1/pi**2)*(integral_eval(kf_p) + integral_eval(kf_n) )#+ cst.m_nucleon*num_density

    return E

def energy_density_leptons(num_density, Y_e, Y_mu):
    kf_e=k_fermi_i(num_density*Y_e)
    kf_mu=k_fermi_i(num_density*Y_mu)

    def integral_eval(kf,m_lepton):
        def integral(k):
            return (k**2)*(sqrt((cst.hbar_c*k)**2 + (m_lepton)**2))
        result,error=quad(integral,0.,kf)
        result=float(result)
        return result
    
    E_leptons=(1/(pi**2))*integral_eval(kf_e,cst.m_electron) + (1/(pi**2))*integral_eval(kf_mu,cst.m_muon)
    
    return E_leptons
    
def energy_density_total(num_density, Y_e, Y_mu):     #nucleonic+leptonic contribution
    Y_p=Y_e+Y_mu
    E_nucleonic=energy_density_rho(num_density, Y_p)
    E_leptonic=energy_density_leptons(num_density, Y_e, Y_mu)
     
    E=E_nucleonic+E_leptonic #+ cst.m_nucleon*num_density
    
    return E

def pressure_rho(num_density, Y_p):         #only nucleonic contribution
    n_p=num_density*Y_p
    n_n=num_density*(1-Y_p)
    
    gss=gss_EOM_numerical_self_int(num_density)
    gww=gww_EOM_self_int(num_density)
    grr=grr_EOM(num_density, Y_p)
    kf_p=k_fermi_i(n_p)
    kf_n=k_fermi_i(n_n)

    def integral_eval(kf):
        def integral(k):
            return ((cst.hbar_c*k)**2)*(k**2)/(sqrt((cst.hbar_c*k)**2 + (cst.m_nucleon-gss)**2))
        result,error=quad(integral,0.,kf)
        result=float(result)
        return result
    
    p=( -(ms_gs**2)*(gss**2) + (mw_gw**2)*(gww**2) + (mr_gr**2)*(grr**2))/(2*(cst.hbar_c**3)) - (1/cst.hbar_c**3)*( (1/3)*b*cst.m_nucleon*gss**3 + (1/4)*c*gss**4 ) + (1/(3*pi**2))*(integral_eval(kf_p) + integral_eval(kf_n) )
    return p

def pressure_leptons(num_density, Y_e, Y_mu):
    kf_e=k_fermi_i(num_density*Y_e)
    kf_mu=k_fermi_i(num_density*Y_mu)
    
    def integral_eval(kf, m_lepton):
        def integral(k):
            return ((cst.hbar_c*k)**2)*(k**2)/(sqrt((cst.hbar_c*k)**2 + (m_lepton)**2))
        result,error=quad(integral,0.,kf)
        result=float(result)
        return result
    
    p_leptons=(1/(3*pi**2))*integral_eval(kf_e, cst.m_electron) + (1/(3*pi**2))*integral_eval(kf_mu, cst.m_muon)
    
    return p_leptons


def pressure_total(num_density, Y_e, Y_mu):       #nucleonic+leptonic contribution
    Y_p=Y_e+Y_mu
    p_nucleonic = pressure_rho(num_density, Y_p)
    p_leptonic= pressure_leptons(num_density, Y_e, Y_mu)
    
    p_total = p_nucleonic + p_leptonic
    
    return p_total

def BE_rho(n, Y_p):
    return (energy_density_rho(n, Y_p))/n -cst.m_nucleon

def chemical_potential(n,Yp,i): #i=0 for neutron, i=1 for proton, i=2 for electron
    g_sigma_sigma=gss_EOM_numerical_self_int(n)
    g_omega_omega=gww_EOM_self_int(n)
    g_rho_rho=grr_EOM(n, Yp)
    nn=n*(1-Yp)
    np=n*Yp
    kfn=cst.hbar_c*k_fermi_i(nn)
    kfp=cst.hbar_c*k_fermi_i(np)
    m_star=cst.m_nucleon-g_sigma_sigma
    
    if i==0:
        return g_omega_omega - 0.5*g_rho_rho + sqrt( kfn**2 + m_star**2 )
    elif i==1:
        return g_omega_omega + 0.5*g_rho_rho + sqrt( kfp**2 + m_star**2 )
    elif i==2:
        return sqrt( kfp**2 + cst.m_electron**2 )
    
def chemical_potential_with_muons(n,Ye,Ymu,i): #i=0 for neutron, i=1 for proton, i=2 for electron
    g_sigma_sigma=gss_EOM_numerical_self_int(n)
    g_omega_omega=gww_EOM_self_int(n)
    g_rho_rho=grr_EOM(n, Ye+Ymu)
    Yp=Ye+Ymu
    nn=n*(1-Yp)
    np=n*(Yp)
    ne=n*Ye
    nmu=n*Ymu
    
    kfn=cst.hbar_c*k_fermi_i(nn)
    kfp=cst.hbar_c*k_fermi_i(np)
    kfe=cst.hbar_c*k_fermi_i(ne)
    kfmu=cst.hbar_c*k_fermi_i(nmu)
    
    m_star=cst.m_nucleon-g_sigma_sigma
    
    if i==0:#neutron
        return g_omega_omega - 0.5*g_rho_rho + sqrt( kfn**2 + m_star**2 )
    elif i==1:#proton
        return g_omega_omega + 0.5*g_rho_rho + sqrt( kfp**2 + m_star**2 )
    elif i==2:#electron
        return sqrt( kfe**2 + cst.m_electron**2 )
    elif i==3:#muon
        return sqrt( kfmu**2 + cst.m_muon**2 )
    
def beta_equilibrium_explicit(n_point, Yp):
    mu_n=chemical_potential(n_point, Yp, 0)
    mu_p=chemical_potential(n_point, Yp, 1)
    mu_e=chemical_potential(n_point, Yp, 2)
    
    return mu_n-mu_p-mu_e

def beta_equilibrium_explicit_muons(n_point,vars ):
    Ye,Ymu=vars
    mu_n=chemical_potential_with_muons(n_point, Ye, Ymu, 0)
    mu_p=chemical_potential_with_muons(n_point, Ye, Ymu, 1)
    mu_e=chemical_potential_with_muons(n_point, Ye, Ymu, 2)
    mu_mu=chemical_potential_with_muons(n_point, Ye, Ymu, 3)

    return [ mu_n-mu_p-mu_e, mu_e-mu_mu] 

# print((cst.m_nucleon-gss_EOM_numerical_self_int(0.153))/cst.m_nucleon)
# print(BE_rho(0.153,0.5))

n_list=linspace(0.01,1.5,1000)

m_star_self_int_list=[] #contains m*/m values
BE_rho_list=[]          #contains binding energy per nucleon
Yp_list=[]              #contains proton fraction at each density point
Ye_list=[]
Ymu_list=[]
Check_euler_list=[]
energy_density_nucleonic_list=[]  #energy density at each density point
pressure_nucleonic_list=[]        #pressure at each density point
energy_density_leptonic_list=[]  
pressure_leptonic_list=[]        
energy_density_total_list=[]
pressure_total_list=[]

Yp_guess=0.001
Ye_guess=0.001
Ymu_guess=0.001
for n in n_list:
    # m_star_self_int_list.append(1-g_sigma_sigma_self_int/cst.m_nucleon)
    
    result=fsolve(lambda Yp: beta_equilibrium_explicit(n,Yp),Yp_guess)  #proton fraction guess set close to 0 hence 0.001
    if chemical_potential(n, result[0], 2)<cst.m_muon:
        #No muons yet
        Set_Yp=(result[0])
        Set_Ye=Set_Yp
        Set_Ymu=0.
        Yp_list.append(Set_Yp)
        Ye_list.append(Set_Ye)
        Ymu_list.append(Set_Ymu)

        #Euler eqn check
        Check_euler_list.append(energy_density_total(n, Set_Ye, Set_Ymu)+pressure_total(n, Set_Ye, Set_Ymu)- (chemical_potential(n, Set_Yp, 0))*(n*(1-Set_Yp)) - (chemical_potential(n, Set_Yp, 1))*(n*Set_Yp) - (chemical_potential(n, Set_Yp, 2))*(n*Set_Yp) )
       
        
    else:
        Set_Ye, Set_Ymu = fsolve(lambda x: beta_equilibrium_explicit_muons(n,x),[Ye_guess, Ymu_guess])  #Electron and Muon fraction guess set
        # print("Muons reached!")
        # print("Density= ",n)
        Set_Yp = Set_Ye + Set_Ymu
        Yp_list.append(Set_Yp)
        Ye_list.append(Set_Ye)
        Ymu_list.append(Set_Ymu)
        
        #Euler eqn check
        Check_euler_list.append(energy_density_total(n, Set_Ye, Set_Ymu)+pressure_total(n, Set_Ye, Set_Ymu)- (chemical_potential_with_muons(n, Set_Ye, Set_Ymu, 0))*(n*(1-Set_Yp)) - (chemical_potential_with_muons(n, Set_Ye, Set_Ymu, 1))*(n*Set_Yp) - (chemical_potential_with_muons(n, Set_Ye,Set_Ymu, 2))*(n*Set_Ye) - (chemical_potential_with_muons(n, Set_Ye,Set_Ymu, 3))*(n*Set_Ymu))

    Yp_guess=Set_Yp         #Setting next guesses close to the last density point's particle fraction values so that fsolve converges faster
    Ye_guess=Set_Ye
    Ymu_guess=Set_Ymu

    BE_rho_list.append(BE_rho(n,Set_Yp))
    
    energy_density_nucleonic_list.append(energy_density_rho(n, Set_Yp))
    energy_density_leptonic_list.append(energy_density_leptons(n, Set_Ye, Set_Ymu))
    energy_density_total_list.append(energy_density_total(n, Set_Ye, Set_Ymu))

    pressure_nucleonic_list.append(pressure_rho(n, Set_Yp))
    pressure_leptonic_list.append(pressure_leptons(n, Set_Ye, Set_Ymu))
    pressure_total_list.append(pressure_total(n, Set_Ye, Set_Ymu))
        
        
    

EOS_array=zeros((len(n_list),3), float)
EOS_array[:,0]=n_list
EOS_array[:,1]=energy_density_total_list
EOS_array[:,2]=pressure_total_list
savetxt('EOS.txt',EOS_array,header='n (fm^-3)            E (Mev/fm^3)             p (Mev/fm^3)')


fig1, ax = plt.subplots(figsize=(6,10))
ax.plot(n_list,BE_rho_list, label="Including isospin forces + Beta equilibrium")
ax.spines['bottom'].set_position(('data', 0)) #shifts X axis to y=0
ax.set_ylim(-5.0,10.0)
ax.set_xlim(0.01,0.5)
ax.set_xlabel(r"$n \mathrm{ (fm}^{-3}\text{)}$")
ax.set_ylabel(r"B/A")
plt.legend()
plt.title("B/A vs kf")
plt.show()

plt.plot(n_list,Yp_list, "r", label="$Y_p$")
plt.plot(n_list,Ye_list, "b",label="$Y_e$")
plt.plot(n_list,Ymu_list, "g", label="$Y_{\mu}$")

plt.title("Particle fraction vs baryon density")
plt.xlabel(r"$n \mathrm{ (fm}^{-3}\text{)}$")
plt.ylabel(r"$Y_i$")
plt.legend()
plt.show()

plt.plot(energy_density_nucleonic_list,pressure_nucleonic_list, label="Nucleonic contribution")
plt.plot(energy_density_leptonic_list,pressure_leptonic_list, label="Leptonic contribution")
plt.plot(energy_density_total_list,pressure_total_list, label="Total")
plt.xlabel(r'$\varepsilon$ (MeV fm$^{-3}$)')
plt.ylabel(r'p (MeV fm$^{-3}$)')
plt.legend()
plt.title(r"P vs $\varepsilon$")
plt.show()

plt.plot(n_list,pressure_nucleonic_list, label="Nucleonic contribution")
plt.plot(n_list,pressure_leptonic_list, label="Leptonic contribution")
plt.plot(n_list,pressure_total_list, label="Total")
plt.xlabel(r'n (fm$^{-3}$)')
plt.ylabel(r'p (MeV fm$^{-3}$)')
plt.legend()
plt.title(r"P vs n")
plt.show()

plt.plot(n_list,energy_density_nucleonic_list, label="Nucleonic contribution")
plt.plot(n_list,energy_density_leptonic_list, label="Leptonic contribution")
plt.plot(n_list,energy_density_total_list, label="Total")
plt.xlabel(r'n (fm$^{-3}$)')
plt.ylabel(r'$\varepsilon$ (MeV fm$^{-3}$)')
plt.legend()
plt.title(r"$\varepsilon$ vs n")
plt.show()

plt.plot(n_list, Check_euler_list)
plt.title("Checking Euler equation")
plt.xlabel("number density (fm^-3)")
plt.ylabel("E + p - mu_n*n_n - mu_p*n_p- mu_e*n_e - (mu_mu*n_mu) ")

print("job done!")