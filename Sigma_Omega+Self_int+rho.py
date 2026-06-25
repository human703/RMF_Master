"""
Questions

1. Better to use analytical forms in integrals?
2. Something wrong with units. In k^2+m*^2 terms, k and m don't have same units
3. Which values of coupling constants to test with? Table 4.4 has values for a sigma-omega-rho model. Can use?
4. Negative pressure? Supposed to happen? But becoming more negative at higher densities
"""

from scipy.optimize import root
from scipy.integrate import quad
from numpy import sqrt, pi, arctanh, linspace
import matplotlib.pyplot as plt

#constants
m_nucleon=939.0#MeV
hbar_c = 197.327#MeV fm 
#hbar_c=1. 
rho_0=0.145#fm^-3

#Initialising fixed parameters
b=0.00561 #dimensionless
#b=0
c=-0.006986 #dimensionless
#c=0

#Walecka parametrization, pg147 in pdf, Theoretical Nuclear and Subnuclear
gs_ms_Walecka=sqrt(267.1/(m_nucleon**2))
gw_mw_Walecka=sqrt(195.9/(m_nucleon**2))
ms_gs_Walecka=1/gs_ms_Walecka
mw_gw_Walecka=1/gw_mw_Walecka

#CONVENTION FOR PARAMETRIZAATION
'''
gs_ms and gw_mw MUST BE in units of MeV^-1. Units are taken care of in the gss function
'''
gs_ms_Glendenning=sqrt(12.684)/hbar_c
gw_mw_Glendenning=sqrt(7.148)/hbar_c
gr_mr_Glendenning=sqrt(4.410)/hbar_c

ms_gs_Glendenning=1/gs_ms_Glendenning
mw_gw_Glendenning=1/gw_mw_Glendenning
mr_gr_Glendenning=1/gr_mr_Glendenning


#Defnining functions
# carefull with momenta
#kf different from pf=hbar*kf
def k_fermi(n): #eq 4.149, Glendenning  
    return (3*(pi**2)*n/2)**(1/3)

def n_fermi(kf):
    return (2*kf**3)/(3*pi**2)

def k_fermi_i(n_i):#n_i is the number density of either neutrons or protons (vector density)
    return (3*(pi**2)*n_i)**(1/3)

def n_fermi_i(kf_i):
    return (kf_i**3)/(3*pi**2)


def gss_EOM_numerical_basic(num_density):   #Outputs the value of g_sigma*sigma at a given value of number density
    def gss(gss):
        kf=k_fermi(num_density)
        m_star=m_nucleon-gss
        def integral_eval(kf):
            def integral(k):
                return (k**2)*(m_star)/(sqrt((hbar_c*k)**2 + (m_star)**2))
            result,error=quad(integral,0,kf)
            result=float(result)
            return result
        return gss-(hbar_c**3)*(gs_ms_Walecka**2)*(2/pi**2)*integral_eval(kf)
    
    x0=[m_nucleon/2] #guess for g_sigma*sigma taken as an intermediate value between 0 to m since it always lies between the two
    solution=root(gss,x0)
    sol=float(solution.x[0])
    return sol

def gss_EOM_numerical_self_int(num_density):   #Outputs the value of g_sigma*sigma at a given value of number density
    def gss(gss):
        kf=k_fermi(num_density)
        m_star=m_nucleon-gss
        def integral_eval(kf):
            def integral(k):
                return (k**2)*(m_star)/(sqrt((hbar_c*k)**2 + (m_star)**2))
            result,error=quad(integral,0,kf)
            result=float(result)
            return result
        return gss-(gs_ms_Glendenning**2)*( (2*(hbar_c**3)/pi**2)*integral_eval(kf) -b*m_nucleon*(gss**2) -c*(gss**3) )
    
    x0=[m_nucleon/2] #guess for g_sigma*sigma taken as an intermediate value between 0 to m since it always lies between the two
    solution=root(gss,x0)
    sol=float(solution.x[0])
    return sol

def gww_EOM_basic(num_density):   #Outputs the value of g_omega*omega_0 at a given value of number density
    return (hbar_c**3)*(gw_mw_Walecka**2)*num_density

def gww_EOM_self_int(num_density):   #Outputs the value of g_omega*omega_0 at a given value of number density
    return (hbar_c**3)*(gw_mw_Glendenning**2)*num_density

def grr_EOM(num_density, Y_p):
    return (hbar_c**3)*(gr_mr_Glendenning**2)*0.5*num_density*(2*Y_p-1)    
    
def energy_density_basic(num_density):
    gss=gss_EOM_numerical_basic(num_density)
    gww=gww_EOM_basic(num_density)
    kf=k_fermi(num_density)
    def integral_eval(kf):
        def integral(k):
            return (k**2)*(sqrt((hbar_c*k)**2 + (m_nucleon-gss)**2))
        result,error=quad(integral,0.,kf)
        result=float(result)
        return result
    
    E=( (ms_gs_Walecka**2)*(gss**2) + (mw_gw_Walecka**2)*(gww**2) )/(2*(hbar_c**3)) + (2/pi**2)*integral_eval(kf)
    return E

def energy_density_self_int(num_density):
    gss=gss_EOM_numerical_self_int(num_density)
    gww=gww_EOM_self_int(num_density)
    kf=k_fermi(num_density)
    def integral_eval(kf):
        def integral(k):
            return (k**2)*(sqrt((hbar_c*k)**2 + (m_nucleon-gss)**2))
        result,error=quad(integral,0.,kf)
        result=float(result)
        return result
    
    E=( (ms_gs_Glendenning**2)*(gss**2) + (mw_gw_Glendenning**2)*(gww**2) )/(2*(hbar_c**3)) + (2/pi**2)*integral_eval(kf) + (1/hbar_c**3)*( (1/3)*b*m_nucleon*gss**3 + (1/4)*c*gss**4 )
    return E

def energy_density_rho(num_density, Y_p):
    n_p=num_density*Y_p
    n_n=num_density*(1-Y_p)
    
    gss=gss_EOM_numerical_self_int(num_density)
    gww=gww_EOM_self_int(num_density)
    grr=grr_EOM(num_density, Y_p)
    kf_p=k_fermi_i(n_p)
    kf_n=k_fermi_i(n_n)

    def integral_eval(kf):
        def integral(k):
            return (k**2)*(sqrt((hbar_c*k)**2 + (m_nucleon-gss)**2))
        result,error=quad(integral,0.,kf)
        result=float(result)
        return result
    
    E=( (ms_gs_Glendenning**2)*(gss**2) + (mw_gw_Glendenning**2)*(gww**2) + (mr_gr_Glendenning**2)*(grr**2))/(2*(hbar_c**3)) + (1/hbar_c**3)*( (1/3)*b*m_nucleon*gss**3 + (1/4)*c*gss**4 ) + (1/pi**2)*(integral_eval(kf_p) + integral_eval(kf_n) )
    return E

def BE_basic(n):
    return (energy_density_basic(n))/n -m_nucleon

def BE_self_int(n):
    return (energy_density_self_int(n))/n -m_nucleon

def BE_rho(n, Y_p):
    return (energy_density_rho(n, Y_p))/n -m_nucleon

number=2.0
print("For rho=", number,"*rho_0")

print("gss")
print("Basic: ", gss_EOM_numerical_basic(rho_0*number), "Self-int: ", gss_EOM_numerical_self_int(rho_0*number))

print("BE")
print("Basic: ", BE_basic(rho_0*number), "Self-int: ", BE_self_int(rho_0*number))
#rho_list=linspace(0.001,55,100) #contains rho/rho_0 values

Set_Yp=0.5         #SET PROTON FRACTION HERE


kf_list=linspace(0.001,4.5,100)
gss_basic_list=[]
gss_self_int_list=[]
#gww_list=[]
m_star_basic_list=[] #contains m*/m values
m_star_self_int_list=[] #contains m*/m values
BE_basic_list=[]
BE_self_int_list=[]
BE_rho_list=[]

for kf in kf_list:
    n=n_fermi(kf)
    # g_sigma_sigma_basic=gss_EOM_numerical_basic(n)
    # gss_basic_list.append(g_sigma_sigma_basic)
    
    # g_sigma_sigma_self_int=gss_EOM_numerical_self_int(n)
    # gss_self_int_list.append(g_sigma_sigma_self_int)
    # #print(len(gss_basic_list))

    # m_star_basic_list.append(1-g_sigma_sigma_basic/m_nucleon)
    # m_star_self_int_list.append(1-g_sigma_sigma_self_int/m_nucleon)

    #gww_list.append(gww_EOM(n))
    BE_basic_list.append(BE_basic(n))
    BE_self_int_list.append(BE_self_int(n))
    BE_rho_list.append(BE_rho(n,Set_Yp))



fig1, ax = plt.subplots(figsize=(6,10))
ax.plot(kf_list,BE_basic_list, label="Basic Sigma-Omega")
ax.plot(kf_list,BE_self_int_list, label="Including Sigma Self-interaction")
ax.plot(kf_list,BE_rho_list, label="Including isospin forces, Yp=0.5")


#plt.axvline(x=kf_list.index(min(BE_basic_list)))

ax.spines['bottom'].set_position(('data', 0)) #shifts X axis to y=0

ax.set_ylim(-20.0,10.0)
ax.set_xlim(0.,2.0)
ax.set_xlabel(r"$k_F \text{ (fm}^{-1}\text{)}$")
ax.set_ylabel(r"B/A")
plt.legend(loc="upper left")
plt.title("B/A vs kf")

#-----------------------------------------------------------------------
# fig2, ax_ = plt.subplots(figsize=(6,10))
# ax_.plot(kf_list,BE_self_int_list)

# ax_.spines['bottom'].set_position(('data', 0)) #shifts X axis to y=0

# #ax_.set_ylim(-18.0,10.0)
# ax_.set_xlim(0.6,1.8)
# ax_.set_xlabel(r"$k_F \text{ (fm}^{-1}\text{)}$")
# ax_.set_ylabel(r"B/A")
# plt.title("B/A vs kf (Self-int)")
# plt.title("B/A vs kf (with self interactions)")


#0-------------------------------------------------------------------------------
'''
fig, ax1 = plt.subplots(figsize=(6, 10))
ax1.set_ylim(0, 1)
ax1.set_xlim(0,4.5)
ax1.spines['bottom'].set_position(('data', 0))
#plt.title("Mean-field theory effective mass")



# Plot Primary Axis: m*/m
ax1.plot(kf_list, m_star_basic_list, label="Basic Sigma-Omega")
ax1.plot(kf_list, m_star_self_int_list, label="Including Sigma self-interaction")

ax1.set_xlabel(r"$k_F \text{ (fm}^{-1}\text{)}$")
ax1.set_ylabel(r"$m^*/m$", fontweight='bold')


ax2 = ax1.twinx()

ax2.plot(kf_list, gss_basic_list, label="Basic Sigma Omega")
ax2.plot(kf_list, gss_self_int_list, label="Including self interactions")
ax2.set_ylim(m_nucleon,0)

ax2.set_ylabel(r"$g_{\sigma} \sigma \text{ (MeV)}$", fontweight='bold')

plt.legend()
plt.title("Mean-field theory effective mass")
'''
print("job done!")
'''


def pressure(num_density):
    gss=gss_EOM_numerical1(num_density)
    gww=gww_EOM(num_density)
    kf=k_fermi(num_density)
    def integral_eval(kf):
        def integral(k):
            return (k**4)/(sqrt((hbar_c*k)**2 + (m_nucleon-gss)**2))
        lim1,lim2=0,kf
        result,error=quad(integral,lim1,lim2)
        result=float(result)
        return result
    
    p=-0.5*(ms_gs**2)*(gss**2) + 0.5*(mw_gw**2)*(gww**2) + (1/3)*(2/pi**2)*integral_eval(kf)
    return p

print(energy_density(0.16),"MeV/fm^3",pressure(0.16),"MeV/fm^3")

'''