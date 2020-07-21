
##############################################################################
#
#      Copyright 2013 Helmholtz-Zentrum Berlin (HZB)
#      Hahn-Meitner-Platz 1
#      D-14109 Berlin
#      Germany
#
#      Author Michael Scheer, Michael.Scheer@Helmholtz-Berlin.de
#
# -----------------------------------------------------------------------
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy (wave_gpl.txt) of the GNU General Public
#    License along with this program.
#    If not, see <http://www.gnu.org/licenses/>.
#
#    Dieses Programm ist Freie Software: Sie koennen es unter den Bedingungen
#    der GNU General Public License, wie von der Free Software Foundation,
#    Version 3 der Lizenz oder (nach Ihrer Option) jeder spaeteren
#    veroeffentlichten Version, weiterverbreiten und/oder modifizieren.
#
#    Dieses Programm wird in der Hoffnung, dass es nuetzlich sein wird, aber
#    OHNE JEDE GEWAEHRLEISTUNG, bereitgestellt; sogar ohne die implizite
#    Gewaehrleistung der MARKTFAEHIGKEIT oder EIGNUNG FueR EINEN BESTIMMTEN ZWECK.
#    Siehe die GNU General Public License fuer weitere Details.
#
#    Sie sollten eine Kopie (wave_gpl.txt) der GNU General Public License
#    zusammen mit diesem Programm erhalten haben. Wenn nicht,
#    siehe <http://www.gnu.org/licenses/>.
#
##############################################################################

#+PATCH,//BRILL/PYTHON
#+DECK,pyBrill,T=PYTHON.

import os,sys,platform,shutil,time

import tkinter as tk
from tkinter import *

import numpy as np
from scipy import special

import matplotlib as mpl
import matplotlib.pyplot as plt

global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
global Nmin, Nmax,Emin,Emax,Bmin,Bmax,FDmin,FDmax,Fmin,Fmax,FBmin,FBmax,FCmin,FCmax
global Mmenu, Omenu, NMmenu, NOmenu, Myfont, Toolbar, Calculated, Fig, Ax, Grid
global Kellip


global \
clight1,cgam1,cq1,alpha1,dnull1,done1,sqrttwopi1,\
emassg1,emasse1,echarge1,emasskg1,eps01,erad1,\
grarad1,hbar1,hbarev1,hplanck1,pol1con1,pol2con1,\
radgra1,rmu01,rmu04pi1,twopi1,pi1,halfpi1,wtoe1,gaussn1,ck934,\
ecdipev,ecdipkev

hbarev1=6.58211889e-16
clight1=2.99792458e8
emasskg1=9.10938188e-31
emasse1=0.510998902e6
emassg1=0.510998902e-3
echarge1=1.602176462e-19
erad1=2.8179380e-15
eps01=8.854187817e-12
pi1=3.141592653589793e0
grarad1=pi1/180.e0
radgra1=180.e0/pi1
hplanck1=6.626176e-34
hbar1=hbarev1*echarge1
wtoe1=clight1*hplanck1/echarge1*1.e9
cq1=55.e0/32.e0/(3.0e0)**0.5*hbar1/emasskg1/clight1
cgam1=4.e0/3.e0*pi1*erad1/emassg1**3
pol1con1=8.e0/5.e0/(3.0e0)**0.5
pol2con1=8.e0/5.e0/(3.0e0)**0.5/2.e0/pi1/3600.e0*emasskg1/hbar1/erad1*emassg1**5

twopi1=2.0e0*pi1
halfpi1=pi1/2.0e0
sqrttwopi1=(twopi1)**0.5
dnull1=0.0e0
done1=1.0e0
rmu01=4.0e0*pi1/1.0e7
rmu04pi1=1.0e-7
alpha1=echarge1**2/(4.0e0*pi1*eps01*hbar1*clight1)
gaussn1=1.0e0/(twopi1)**0.5

ck934=echarge1/(2.0e0*pi1*emasskg1*clight1)/100.0e0
def fqnke(n,K,Kyx):

    #NOTE: x and y reversed compared to brill_ellip.kumac

    # n is integer order of Fn(K)
    nm = int((n-1)/2)
    np = int((n+1)/2)

    K2 = K*K
    Kx2 = K2 / (1.0+Kyx**2)
    Ky2 = K2 - Kx2
    Kx = sqrt(Kx2)
    Ky = 0.0
    if Ky2 > 0.0: Ky = sqrt(Ky2)

    K221 = 1. + K2/2.

    x = n*(Kx2-Ky2) / (4.*K221)

    Jm = special.jv(nm,x)
    Jp = special.jv(np,x)

    Ax = Kx * (Jp-Jm)
    Ay = Ky * (Jp+Jm)

    fnke = (n/K221)**2 * (Ax*Ax+Ay*Ay)

    qnke = K221 * fnke / n

    return fnke,qnke

#enddef fqnk(n,K)

def fqnk(n,K):

    # n is integer order of Fn(K)
    nm = int((n-1)/2)
    np = int((n+1)/2)

    K2 = K*K
    K221 = 1. + K2/2.

    x = n*K2 / (4.*K221)

    Jm = special.jv(nm,x)
    Jp = special.jv(np,x)

    fnk = n * K / K221 * (Jm - Jp)
    fnk = fnk * fnk

    qnk = K221 * fnk / n

    return fnk,qnk

#enddef fqnk(n,K)

def calc_brill(l='?', nKvals=101, Kmin=0.5, Kmax=3., n=100, ebeam=1.722, curr=0.1,
          emitx=4.4, emity=0.066, betx=14., bety=3.4,
          sige=0.001, mode=1):

    global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList

    global Calculated

    if type(l) == str and l == '?' :
        print("calc_brill(l='?', nKvals=101,Kmin=0.5, Kmax=3., n=100, ebeam=1.722, curr=0.1,")
        print("emitx=4.4, emity=0.066, betx=14., bety=3.4,sige=0.001, mode=1)")
        return

    dK = (Kmax-Kmin)/(nKvals-1)
    Kvals = np.arange(Kmin,Kmax+dK,dK)
    b0 = Kvals/(echarge1 * L /1000./(2.*pi1*emasskg1*clight1))

    if type(KyxList) != int:
        print("\n***********************************************************")
        print("Attention: For elliptical undulators K is shift-dependend!")
        print("Thus K must actually set for each harmonic...\n")
        print("***********************************************************\n")
        time.sleep(3)
    #endif type(KyxList) != int

    Fkn = []
    Qn = []
    F = []
    FB = []
    FC = []
    FD = []
    B = []
    Lam = []
    Harm = []
    Sigr = []
    Sigrp = []

    for k in range(12):

        i = k
        if i == 0: i = 1

        if type(KyxList) == int:
            fk, q  = fqnk(i,Kvals)
        else:
            fk, q  = fqnke(i,Kvals,KyxList[i])
        #endif type(KyxList) == int:

        Fkn.append(fk)
        Qn.append(q)

        F.append(0.0)
        FB.append(0.0)
        FC.append(0.0)
        FD.append(0.0)
        B.append(0.0)
        Lam.append(0.0)
        Harm.append(0.0)
        Sigr.append(0.0)
        Sigrp.append(0.0)
    #endfor k in range(10)

    #for i in [1,3,5,7,9,11]

    sigx=np.sqrt(emitx*1.e-9*betx)*1000. #mm
    sigxp=np.sqrt(emitx*1.e-9/betx)*1000. #mrad
    sigy=np.sqrt(emity*1.e-9*bety)*1000.
    sigyp=np.sqrt(emity*1.e-9/bety)*1000.

    if mode == 1:
        print("\n  Mode=1 (Kim):")
        print("\n  sigr := sqrt(lambda*length)/4/pi")
        print("  sigrp := sqrt(lambda/length)")
        print("  sigr*sigrp := lambda/(4*pi)\n\n")
    elif mode == 2:
        print("\n  Mode=2 (Walker, recommended):")
        print("\n  sigr := sqrt(2*lambda*length)/2/pi")
        print("  sigrp := sqrt(lambda/2/length)")
        print("  sigr*sigrp := lambda/(2*pi)\n\n")
    elif mode == -1:
        print("\n  Mode=-1 Erik Wallen, MAX-IV:")
        print("\n  sigw[i]=0.36/[i]/[N] | width of harmonic")
        print("  mulam[i]=sqrt(1.+(2.*[sigE]*[i]*[N]/0.36)**2)")
        print("  sigr[i]=sqrt(lam1/[i]*[N]*[l]/(8.*pi**2))") # like Kim
        print("  sigrp[i]=sqrt(mulam[i]*lam1/[i]/(2.*[N]*[l]))") # rad like Walker
        print("  sigr*sigrp := lambda/(4*pi)\n\n") # like Kim
    elif mode == 3:
        print("\n  Mode=3:")
        print("\n  sigr := sqrt(sqrt(2)*lambda*length)/2/pi")
        print(" 	sigrp := sqrt(lambda/2/sqrt(2)/length)")
        print(" 	sigr*sigrp := lambda/sqrt(2)/(2*pi)\n\n")
    else:
        print("\n *** Mode has no meaning, set to 1 (Kim) *** \n")
        mode = 1
    #endif

    time.sleep(1)

    if not Calculated:
        print('\n For the flux calculation the enery-spread is not taken into account, since')
        print(' it has no effect for the brillant flux and a smaller effect for the max. flux')
        print(' compared to the effect on the flux-density.')
        print(" The flux is always calculated according to Kim's formular (xray")
        print(' data booklet eqn. 17). This overestimates the flux by a factor')
        print(' of two, but since the max. is about a this factor higher than')
        print(' the on-resonant flux, this seems to be alright.\n\n')
        Calculated = True

    time.sleep(1)

    pi = pi1

    for i in [1,3,5,7,9,11]:

        lami = 13.056*(1.+Kvals**2/2.)*L/10./ebeam**2/1.e10*1.e3/i # mm
        Lam[i] = lami

        harmi = i*0.950/(1.+Kvals**2/2.)/(l/10.)*ebeam**2 # keV
        Harm[i] = harmi

        rsigphi = \
        (1./(i*n*2.*np.sqrt(2.0))) / \
        np.sqrt((1.0/(i*n*2.*np.sqrt(2.0)))**2+(2.0*sige)**2)

        if mode == 1:
            sigrpi = np.sqrt(lami/(n*l)) # rad
            sigri = lami/4./pi/sigrpi # mm
        elif mode == 2:
            sigrpi = np.sqrt(lami/2./(n*l)) # rad
            sigri = lami/2./pi/sigrpi # mm
        elif mode == -1:
            #MAX IV formulas (Erik Wallen?)
            sigwi = 0.36/i/n # width of harmonic
            mulami = np.sqrt(1.+(2.*sigE*i*n/0.36)**2)
            rsigphi = 0.
            sigri = np.sqrt(lam1/i*n*l/(8.*pi**2)) # like Kim
            sigrpi = np.sqrt(mulami*lam1/i/(2.*n*l)) # rad # like Walker

        elif mode == 3:
            sigrpi = np.sqrt(lami/(n*l*2.*np.sqrt(2.))) #rad
            sigri=np.sqrt(np.sqrt(2.)*lami*(n*l))/2./pi #mm
        #endif mode == ...

        Sigr[i] = sigri
        Sigrp[i] = sigrpi

        sigrpi=sigrpi*1000 # mrad

        sigrxi = np.sqrt(sigx**2+sigri**2)
        sigrpxi = np.sqrt(sigxp**2+sigrpi**2)
        sigryi = np.sqrt(sigy**2+sigri**2)
        sigrpyi = np.sqrt(sigyp**2+sigrpi**2)

        rsigxyi = sigrpi**2/(sigrpxi*sigrpyi)

        F[i] = 1.431e14 * n * Qn[i] * curr # flux, data booklet Kim
        FD[i] = 1.744e14 * n**2 * ebeam**2 * curr * Fkn[i] # spectral brightness, data booklet Kim
        FB[i] = FD[i]*2. * pi * sigrpi**2 # brilliant flux
        FD[i] = FD[i]*rsigphi*rsigxyi # d.h. keine Emittanzwirkung auf FB!

        if mode == 1:
            B[i] = F[i] * rsigphi / (2.*pi)**2 / (sigrxi*sigrpxi*sigryi*sigrpyi)
        elif mode == 2:
            B[i] = FB[i] * rsigphi / (2.*pi)**2 /(sigrxi*sigrpxi*sigryi*sigrpyi)
        elif mode == -1:
            # MAX IV formulas (Erik Wallen?)
            rsigxyi = sigrpi**2 / (sigrpxi*sigrpyi)
            F[i] = 1.431e14 * n * Qn[i] * curr/2. # flux, data booklet Kim divived by 2 (MAX IV formula)
            FB[i] = 1.431e14 * n * Qn[i] * curr/2. # flux, data booklet Kim divived by 2 (MAX IV formula)
            B[i] = FB[i] / (2.*pi)**2 / (sigrxi*sigrpxi*sigryi*sigrpyi)
        elif mode == 3:
            B[i] = FB[i] * rsigphi/(2.*pi)**2 / (sigrxi*sigrpxi*sigryi*sigrpyi)
        #endif mode

        FC[i]= B[i] * (lami/2.)**2 * 1.0e6 # mrad**2 -> rad**2!!

    #endfor i in [1,3,5,7,9,11]

#enddef calc_brill(l='?', n=100, ebeam=1.722, curr=0.1,

# Wrting
def write_brill(fout="brilliance.dat"):

    global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList

    Fout = open(fout,'w')
    Fout.write("* n, Harm, Flux-dens., Flux, Brilliance, Fbrill, Fcoh, SigR, SigRp\n")

    ll = len(Harm[1]) - 1

    for i in [1,3,5,7,9,11]:
        for l in range(len(Harm[i])):
            k = ll - l
            sw = str(i)
            sw += " " + '{:.4g}'.format(Harm[i][k])
            sw += " " + '{:.4g}'.format(FD[i][k])
            sw += " " + '{:.4g}'.format(F[i][k])
            sw += " " + '{:.4g}'.format(B[i][k])
            sw += " " + '{:.4g}'.format(FB[i][k])
            sw += " " + '{:.4g}'.format(FC[i][k])
            sw += " " + '{:.4g}'.format(Sigr[i][k])
            sw += " " + '{:.4g}'.format(Sigrp[i][k])
            sw += "\n"
            Fout.write(sw)
        #endfor k in range(len(F)))
    #endfor i in [1,3,5,7,9,11]

    Fout.close()
#enddef write_brill()

def _writelastrun():

    global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global L, nKvals, Kvals, b0, Kmin, Kmax, Kellip, N, Ebeam, Curr, \
    Emitx, Emity, Betx, Bety, Sige, Mode
    global Mmenu, Myfont
    global SetUp, Vsetup, LastSetup

    _UpdateVars()

    fl = ".pybrill_last.dat"

    Fl = open(fl,"w")

    Fl.write( str(L) + "\n")
    Fl.write( str(nKvals) + "\n")
    Fl.write( str(Kmin) + "\n")
    Fl.write( str(Kmax) + "\n")
    Fl.write( str(Kellip) + "\n")
    Fl.write( str(Nmin) + "\n")
    Fl.write( str(Nmax) + "\n")
    Fl.write( str(N) + "\n")
    Fl.write( str(Ebeam) + "\n")
    Fl.write( str(Curr) + "\n")
    Fl.write( str(Emitx) + "\n")
    Fl.write( str(Emity) + "\n")
    Fl.write( str(Betx) + "\n")
    Fl.write( str(Bety) + "\n")
    Fl.write( str(Sige) + "\n")
    Fl.write( str(Mode) + "\n")

    Fl.close()

#enddef _writelastrun()

def _readlastrun():

    global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global L, nKvals, Kvals, b0, Kmin, Kmax, Kellip, N, Ebeam, Curr, \
    Emitx, Emity, Betx, Bety, Sige, Mode
    global Mmenu, Myfont
    global SetUp, Vsetup, LastSetup

    fl = ".pybrill_last.dat"

    if os.path.exists(fl):
        Fl = open(fl,"r")

        line = Fl.readline().strip()
        L = float(line)
        line = Fl.readline().strip()
        nKvals = int(line)
        line = Fl.readline().strip()
        Kmin = float(line)
        line = Fl.readline().strip()
        Kmax = float(line)
        line = Fl.readline().strip()
        Kellip = False
        if line.lower() in ['true','1','yes','y','j','ja']: Kellip = True
        line = Fl.readline().strip()
        Nmin = int(line)
        line = Fl.readline().strip()
        Nmax = int(line)
        line = Fl.readline().strip()
        N = int(line)
        line = Fl.readline().strip()
        Ebeam = float(line)
        line = Fl.readline().strip()
        Curr = float(line)
        line = Fl.readline().strip()
        Emitx = float(line)
        line = Fl.readline().strip()
        Emity = float(line)
        line = Fl.readline().strip()
        Betx = float(line)
        line = Fl.readline().strip()
        Bety = float(line)
        line = Fl.readline().strip()
        Sige = float(line)
        line = Fl.readline().strip()
        Mode = int(line)

        Fl.close()
    #endif os.path.exists(".pybrilllast.dat")

#enddef _readlastrun

def pwplot(y,n,ftit):

    global L, nKvals, Kvals, b0, Kmin, Kmax, Kellip, N, Ebeam, Curr, Emitx, Emity
    global Betx, Bety, Sige, Mode, KyzList

    x = Harm[n]
    plt.plot(x,y)

    fn = ftit + "_Harm_" + str(n) + ".dat"

    Fn = open(fn,'w')

    Fn.write("* Period-length " + str(L) + "\n")
    Fn.write("* Beam energy [GeV], Curr [A] " + str(Ebeam) + " " + str(Curr) + "\n")
    Fn.write("* Hori. and vert. Emittance [nm-rad] " +  str(Emitx) + " " + str(Emity) + "\n")
    Fn.write("* Hori. and vert. Beta-functions [m] " + str(Betx) + " " + str(Bety) + "\n")
    Fn.write("* Rel. beam energy spread " + str(Sige) + "\n")

    for j in range(nKvals):
        i = nKvals - j - 1
        line = str(j+1) + " " + str(x[i]) + " " + str(y[i])
        line +=  " " + str(Kvals[i]) + " " + str(b0[i])
        Fn.write(line + "\n")
    Fn.close()

    print("Written to " + fn)
#enddef pwplot(x,y,fn='pyBrill.dat')

def grid(alpha=0.5):
    global Fig, Ax, Grid
    if Grid:
        plt.grid(Grid)
        Ax.grid(which='minor',alpha=alpha)
    #endif Grid
#enddef grid()

def zoom(xmin,xmax,ymin,ymax):
    global Fig, Ax
    Ax.axis([xmin,xmax,ymin,ymax])

    grid()
    plt.show(block=False)
#enddef zoom()

def _pcohflux():

    global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Nmin, Nmax,Emin,Emax,Bmin,Bmax,Fdmin,Fdmax,Fmin,Fmax,FBmin,FBmax,Fcmin,Fcmax
    global Fig,Ax, Curr

    if Calculated == False: _calc()

    Fig = plt.gcf()
    Fig.clear()

    Nmin = max(Nmin,1)
    Nmax = min(Nmax,11)

    for i in [1,3,5,7,9,11]:
        if i >= Nmin and i <= Nmax:
            pwplot(FC[i],i,'Coh_Flux')

    Fig = plt.gcf()

    Ax = plt.gca()
    Ax.set_xscale('log')
    Ax.set_yscale('log')

    Ax.set_title("Coherent Flux")

    Ax.set_xlabel("photon enery [keV]")
    Ax.set_ylabel("N [1/s/mm$^{2}$/0.1%BW/" + str(Curr*1000.) + "mA]")

    if FCmin == -1. and FCmax == -1.:
        ymin, ymax = Ax.get_ylim()
        Ax.set_ylim(ymax/1.e3,ymax)
    #endif FCmin == -1. and FCmax == -1.:

    grid()
    plt.show(block=False)
#enddef _pcohflux()

def _pbrillflux():

    global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Nmin, Nmax,Emin,Emax,Bmin,Bmax,Fdmin,Fdmax,Fmin,Fmax,FBmin,FBmax,Fcmin,Fcmax
    global Fig,Ax, Curr

    if Calculated == False: _calc()

    Fig = plt.gcf()
    Fig.clear()

    Nmin = max(Nmin,1)
    Nmax = min(Nmax,11)

    for i in [1,3,5,7,9,11]:
        if i >= Nmin and i <= Nmax:
            pwplot(FB[i],i,'Brilliant_Flux')

    Fig = plt.gcf()

    Ax = plt.gca()
    Ax.set_xscale('log')
    Ax.set_yscale('log')

    Ax.set_title("Brilliant Flux")

    Ax.set_xlabel("photon enery [keV]")
    Ax.set_ylabel("N [1/s/mm$^{2}$/0.1%BW/" + str(Curr*1000.) + "mA]")

    if FBmin == -1. and FBmax == -1.:
        ymin, ymax = Ax.get_ylim()
        Ax.set_ylim(ymax/1.e4,ymax)
    #endif FCmin == -1. and FCmax == -1.:

    grid()
    plt.show(block=False)
#enddef _pbrillflux()

def _pflux():

    global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Nmin, Nmax,Emin,Emax,Bmin,Bmax,Fdmin,Fdmax,Fmin,Fmax,FBmin,FBmax,Fcmin,Fcmax
    global Fig,Ax, Curr

    if Calculated == False: _calc()

    Fig = plt.gcf()
    Fig.clear()

    Nmin = max(Nmin,1)
    Nmax = min(Nmax,11)

    for i in [1,3,5,7,9,11]:
        if i >= Nmin and i <= Nmax:
            pwplot(F[i],i,'Flux')

    Fig = plt.gcf()

    Ax = plt.gca()
    Ax.set_xscale('log')
    Ax.set_yscale('log')

    Ax.set_title("Flux")

    Ax.set_xlabel("photon enery [keV]")
    Ax.set_ylabel("N [1/s/0.1%BW/" + str(Curr*1000.) + "mA]")

    if Fmin == -1. and Fmax == -1.:
        ymin, ymax = Ax.get_ylim()
        Ax.set_ylim(ymax/1.e4,ymax)
    #endif FCmin == -1. and FCmax == -1.:

    grid()
    plt.show(block=False)
#enddef _pflux()

def _pfluxden():

    global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Nmin, Nmax,Emin,Emax,Bmin,Bmax,Fdmin,Fdmax,Fmin,Fmax,FBmin,FBmax,Fcmin,Fcmax
    global Fig,Ax, Curr

    if Calculated == False: _calc()

    Fig = plt.gcf()
    Fig.clear()

    Nmin = max(Nmin,1)
    Nmax = min(Nmax,11)

    for i in [1,3,5,7,9,11]:
        if i >= Nmin and i <= Nmax:
            pwplot(FD[i],i,'Flux-density')

    Fig = plt.gcf()

    Ax = plt.gca()
    Ax.set_xscale('log')
    Ax.set_yscale('log')

    Ax.set_title("Flux-density")

    Ax.set_xlabel("photon enery [keV]")
    Ax.set_ylabel("N [1/s/mrad$^{2}$/0.1%BW/" + str(Curr*1000.) + "mA]")

    if FDmin == -1. and FDmax == -1.:
        ymin, ymax = Ax.get_ylim()
        Ax.set_ylim(ymax/1.e4,ymax)
    #endif FCmin == -1. and FCmax == -1.:

    grid()
    plt.show(block=False)
#enddef _pfluxden()

def _pbrill():

    global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Nmin, Nmax,Emin,Emax,Bmin,Bmax,Fdmin,Fdmax,Fmin,Fmax,FBmin,FBmax,Fcmin,Fcmax
    global Fig,Ax, Curr, Vsetup

    if Calculated == False: _calc()

    Fig = plt.gcf()
    Fig.clear()

    Nmin = max(Nmin,1)
    Nmax = min(Nmax,11)

    for i in [1,3,5,7,9,11]:
        if i >= Nmin and i <= Nmax:
            pwplot(B[i],i,'Brilliance')

    Fig = plt.gcf()

    Ax = plt.gca()
    Ax.set_xscale('log')
    Ax.set_yscale('log')

    Ax.set_title("Brilliance")

    Ax.set_xlabel("photon enery [keV]")
    Ax.set_ylabel("N [1/s/mm$^{2}$/mrad$^{2}$/0.1%BW/" + str(Curr*1000.) + "mA]")

    if Bmin == -1. and Bmax == -1.:
        ymin, ymax = Ax.get_ylim()
        Ax.set_ylim(ymax/1.e4,ymax)
    #endif FCmin == -1. and FCmax == -1.:

    grid()
    plt.show(block=False)
#enddef _pbrill()

def _UpdateVars():

    global Vsetup, Nmin, Nmax, Kellip, KyxList
    global L, nKvals, Kvals, b0, Kmin, Kmax, Kellip, N, Ebeam, Curr, Emitx, Emity, Betx, Bety, Sige, Mode

    L = float(Vsetup[0][1])
    nKvals = int(float(Vsetup[1][1]))
    Kmin = float(Vsetup[2][1])
    Kmax = float(Vsetup[3][1])
    Kellip = Vsetup[4][1]
    if type(Kellip) == str:
        if Kellip.lower() in ['true','1','yes','y','j','ja']: Kellip = True
        else: Kellip = False
    else: Kellip = bool(Vsetup[4][1])
    Nmin = int(float(Vsetup[5][1]))
    Nmax = int(float(Vsetup[6][1]))
    N = int(float(Vsetup[7][1]))
    Ebeam = float(Vsetup[8][1])
    Curr = float(Vsetup[9][1])
    Emitx = float(Vsetup[10][1])
    Emity = float(Vsetup[11][1])
    Betx = float(Vsetup[12][1])
    Bety = float(Vsetup[13][1])
    Sige = float(Vsetup[14][1])
    Mode = int(float(Vsetup[15][1]))

    if Kellip: KyxList = [1.0, 0.42, 0.32, 0.27, 0.24, 0.22]
    else: KyxList = 0

    dK = (Kmax-Kmin)/(nKvals-1)
    Kvals = np.arange(Kmin,Kmax+dK,dK)
    b0 = Kvals/(echarge1 * L /1000./(2.*pi1*emasskg1*clight1))

#enddef _UpdateVars()

def _calc():
    global Calculated, LastSetup, Kellip
    global L, nKvals, Kvals, b0, Kmin, Kmax, Kellip, N, Ebeam, Curr, Emitx, Emity, Betx, Bety, Sige, Mode

    Calculated = True
    _UpdateVars()

    calc_brill(\
    L, nKvals, Kmin, Kmax, N, Ebeam, Curr, \
    Emitx, Emity, Betx, Bety, Sige, Mode \
    )

    #if type(LastSetup) == int:
    #    print("\n",Vsetup,"\n")

#enddef calc

def _exit():

    try: _writelastrun()
    except: pass

    if platform.system() == 'Windows':
        stat = os.system("taskkill /F /PID " + str(os.getpid()))
    else:
        stat = os.system("kill " + str(os.getpid()))
    #endif platform.system() == 'Windows'
#enddef _exit()

def _SetUpIn(event,kvar):
    global LastSetup
    LastSetup = [event,kvar]
#enddef _SetUpOut(event,kvar)

def _SetUpOut(event,kvar):
    global Vsetup, LastSetup

    ev = LastSetup[0].widget
    val = ev.get()
    Vsetup[kvar][1] = val

    if kvar == 4:
        v = Vsetup[4][1]
        if type(v) == str:
            if v.lower() in ['true','1','yes','y','j','ja']: Kellip = True
            else: Kellip = False
        else: Kellip = bool(Kellip)

#enddef _SetUpOut(event,kvar)

def _closeSetUp():
    global Vsetup,LastSetup, Kellip, KyxList

    if LastSetup:
        ev = LastSetup[0].widget
        kvar = LastSetup[1]
        val = ev.get()
        Vsetup[kvar][1] = val
    #endif LastSetup

    v = Vsetup[4][1]
    if type(v) == str:
        if v.lower() in ['true','1','yes','y','j','ja']: Kellip = True
        else: Kellip = False
    else: Kellip = bool(Kellip)

    SetUp.destroy()
#def _closeSetUp(win)

def _setup():

    global Fkn, F,FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global L, nKvals, Kvals, b0, Kmin, Kmax, Kellip, N, Ebeam, Curr, \
    Emitx, Emity, Betx, Bety, Sige, Mode
    global Mmenu, Myfont
    global SetUp, Vsetup, LastSetup

    SetUp = Toplevel()
    LastSetup = 0

    for i in range(len(Vsetup)):
        f = Frame(SetUp)
        flab = Label(f,text=Vsetup[i][0])
        fent =  Entry(f)
        fent.insert(1,Vsetup[i][1])
        flab.pack(side=LEFT)
        fent.pack(side=RIGHT)
        fent.bind('<FocusIn>',lambda event,kvar=i:_SetUpIn(event,kvar))
        fent.bind('<FocusOut>',lambda event,kvar=i:_SetUpOut(event,kvar))
        f.pack(fill='x')
    #ENDfor i in range(len(V))

    bClose = Button(SetUp,text='Ok',command=_closeSetUp)
    bClose.pack()

    v = Vsetup[4][1]
    if type(v) == str:
        if v.lower() == ['true','1','yes','ja']: Kellip = True
        else: Kellip = False
    else: Kellip = bool(Kellip)

    if Kellip:
        print("\n***********************************************************")
        print("Attention: For elliptical undulators K is shift-dependend!")
        print("Thus K must actually set for each harmonic...")
        print("***********************************************************\n")
    #endif Kellip

#enddef _setup

def getgeo():

    fig = plt.gcf()
    geo=fig.canvas.manager.window.wm_geometry()
    git = geo.split('+')
    wh = git[0].split('x')
    x = int(git[1]); y = int(git[2])
    w = int(wh[0]); h = int(wh[1]);

    return w,h,x,y
#def getgeo(w,h,x,y)

def _showMenu(menu):

    global Mmenu, Omenu, Toolbar, NMmenu, NOmenu, Myfont

    fontsize = int(Myfont[1])
    w,h,x,y = getgeo()

    menu.post(x+int(w*0.35),y+h-2*fontsize*(NMmenu+1))
    #endif menu == Omenu

#enddef _showMenu()

def _canbutbrill(ev):
    global Mmenu, Omenu, NMmenu, NOmenu
    print("_canbutbrill",ev)
#enddef _canbutbrill(ev)

#######################################################

print("\n\n--------------------------------------------------------------------------")
print("--------------------------------------------------------------------------")
print("\n\n Mode=1 (Kim):")
print("-------------------\n")
print("\n sigr := sqrt(lambda*length)/4/pi")
print(" sigrp := sqrt(lambda/length)")
print(" sigr*sigrp := lambda/(4*pi)\n\n")

print("\n Mode=2 (Walker, recommended):")
print("------------------------------\n")
print("\n sigr := sqrt(2*lambda*length)/2/pi")
print(" sigrp := sqrt(lambda/2/length)")
print(" sigr*sigrp := lambda/(2*pi)\n\n")

print("\n Mode=-1 Erik Wallen, MAX-IV:")
print("-----------------------------\n")
print("\n sigw[i]=0.36/[i]/[N] | width of harmonic")
print(" mulam[i]=sqrt(1.+(2.*[sigE]*[i]*[N]/0.36)**2)")
print(" sigr[i]=sqrt(lam1/[i]*[N]*[l]/(8.*pi**2))") # like Kim
print(" sigrp[i]=sqrt(mulam[i]*lam1/[i]/(2.*[N]*[l]))") # rad like Walker
print(" sigr*sigrp := lambda/(4*pi)\n\n") # like Kim

print("\n Mode=3 (WAVE):")
print("---------------\n")
print("\n sigr := sqrt(sqrt(2)*lambda*length)/2/pi")
print(" sigrp := sqrt(lambda/2/sqrt(2)/length)")
print(" sigr*sigrp := lambda/sqrt(2)/(2*pi)\n\n")

print("-----------------------------------------------------------------------\n")

print('\n For the flux calculation the enery-spread is not taken into account, since')
print(' it has no effect for the brillant flux and a smaller effect for the max. flux')
print(' compared to the effect on the flux-density.')
print(" The flux is always calculated according to Kim's formular (xray")
print(' data booklet eqn. 17). This overestimates the flux by a factor')
print(' of two, but since the max. is about a this factor higher than')
print(' the on-resonant flux, this seems to be alright.\n\n')

print("-----------------------------------------------------------------------\n")

L = 50.
nKvals = 101
Kmin = 0.5
Kmax = 3.
Kellip = False
Nmin = 1
Nmax = 11
N = 100
Ebeam = 1.722
Curr = 0.1
Emitx = 4.4
Emity = 0.066
Betx = 14.
Bety = 3.4
Sige = 0.001
Mode = 2 # Walker

_readlastrun()

Vsetup = []
Vsetup.append(["Period-length [mm]",L])
Vsetup.append(["Number of K values",nKvals])
Vsetup.append(["Kmin",Kmin])
Vsetup.append(["Kmax",Kmax])
Vsetup.append(["Elliptical Undulator",Kellip])
Vsetup.append(["Lowest harmonic",Nmin])
Vsetup.append(["Highest harmonic",Nmax])
Vsetup.append(["Number of periods",N])
Vsetup.append(["Beam energy [GeV]",Ebeam])
Vsetup.append(["Current [A]",Curr])
Vsetup.append(["Hor. Emit. [nm-rad]",Emitx])
Vsetup.append(["Ver. Emit. [nm-rad]",Emity])
Vsetup.append(["Hori. Beta function",Betx])
Vsetup.append(["Vert. Beta function",Bety])
Vsetup.append(["Rel. energy spread",Sige])
Vsetup.append(["Mode [-1,1,2,3]",Mode])

Fkn = []
F = []
FD = []
FC = []
FB = []
Qn = []
B = []

Lam = []
Harm = []
Sigr = []
Sigrp = []

Calculated = False

Mmenu = 0
Omenu = 0
NMmenu = 0
NOmenu = 0

Emin = -1.
Emax = -1.

Bmin = -1.
Bmax = -1.
Fmin = -1.
Fmax = -1.
FDmin = -1.
FDmax = -1.
FCmin = -1.
FCmax = -1.
FBmin = -1.
FBmax = -1.

KyxList = 0

Grid = True
LastSetup = 0

mpl.use('TkAgg')

Fig = plt.figure()
Fig.show()

#Ax = plt.gca()
#Fig.canvas.mpl_disconnect(CanButId)

Wmain = plt.gcf()
Wmaster = Wmain.canvas.toolbar.master

Toolbar = Wmain.canvas.toolbar
Myfont = ('arial',13)

CanButBrill = Wmain.canvas.mpl_connect('button_press_event',_canbutbrill)

Mmenu = Menu(Toolbar,tearoff=1,font=Myfont)
mPlot = Menu(Mmenu,tearoff=1,font=Myfont)

##########
NMmenu += 1
Mmenu.add_command(label='Set Up', command=_setup)

NMmenu += 1
Mmenu.add_command(label='Calculate', command=_calc)

NMmenu += 1
Mmenu.add_cascade(label='Plot', menu=mPlot)

NMmenu += 1
Mmenu.add_command(label='Exit', command=_exit)
##########

mPlot.add_command(label='Brilliance', command=_pbrill)
mPlot.add_command(label='Flux-density', command=_pfluxden)
mPlot.add_command(label='Flux', command=_pflux)
mPlot.add_command(label='Coherent Flux', command=_pcohflux)
mPlot.add_command(label='Brilliant Flux', command=_pbrillflux)

##########

bMmenu = Button(Toolbar,text='Menu',font=Myfont,
                command= lambda menu = Mmenu: _showMenu(menu))

bMmenu.pack(side=LEFT)

_setup()
_pbrill()
