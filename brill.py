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

# +PATCH,//BRILL/PYTHON
# +DECK,pyBrill,T=PYTHON.

import os, sys, platform, shutil, time

import tkinter as tk
from tkinter import *

import numpy as np
from scipy import special

import matplotlib as mpl
import matplotlib.pyplot as plt

global Fkn, F, FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
global Nmin, Nmax, Emin, Emax, Bmin, Bmax, FDmin, FDmax, Fmin, Fmax, FBmin, FBmax, FCmin, FCmax
global Mmenu, Omenu, NMmenu, NOmenu, Myfont, Toolbar, Calculated, Fig, Ax, Grid
global Kellip


global clight1, cgam1, cq1, alpha1, dnull1, done1, sqrttwopi1, emassg1, emasse1
global echarge1, emasskg1, eps01, erad1, grarad1, hbar1, hbarev1, hplanck1, pol1con1
global pol2con1, radgra1, rmu01, rmu04pi1, twopi1, pi1, halfpi1, wtoe1, gaussn1, ck934
global ecdipev, ecdipkev

hbarev1 = 6.58211889e-16
clight1 = 2.99792458e8
emasskg1 = 9.10938188e-31
emasse1 = 0.510998902e6
emassg1 = 0.510998902e-3
echarge1 = 1.602176462e-19
erad1 = 2.8179380e-15
eps01 = 8.854187817e-12
pi1 = 3.141592653589793e0
grarad1 = pi1 / 180.0e0
radgra1 = 180.0e0 / pi1
hplanck1 = 6.626176e-34
hbar1 = hbarev1 * echarge1
wtoe1 = clight1 * hplanck1 / echarge1 * 1.0e9
cq1 = 55.0e0 / 32.0e0 / (3.0e0) ** 0.5 * hbar1 / emasskg1 / clight1
cgam1 = 4.0e0 / 3.0e0 * pi1 * erad1 / emassg1 ** 3
pol1con1 = 8.0e0 / 5.0e0 / (3.0e0) ** 0.5
pol2con1 = pol1con1 / 2.0e0 / pi1 / 3600.0e0 * emasskg1 / hbar1 / erad1 * emassg1 ** 5

twopi1 = 2.0e0 * pi1
halfpi1 = pi1 / 2.0e0
sqrttwopi1 = (twopi1) ** 0.5
dnull1 = 0.0e0
done1 = 1.0e0
rmu01 = 4.0e0 * pi1 / 1.0e7
rmu04pi1 = 1.0e-7
alpha1 = echarge1 ** 2 / (4.0e0 * pi1 * eps01 * hbar1 * clight1)
gaussn1 = 1.0e0 / (twopi1) ** 0.5

ck934 = echarge1 / (2.0e0 * pi1 * emasskg1 * clight1) / 100.0e0


def fqnke(n: np.ndarray, K: np.ndarray, Kyx: np.ndarray):
    """NOTE: x and y reversed compared to brill_ellip.kumac. n is integer order of Fn(K)"""
    nm = n - 1 // 2
    np = (n + 1) // 2

    K2 = K * K
    Kx2 = K2 / (1.0 + Kyx ** 2)
    Ky2 = K2 - Kx2
    Kx = sqrt(Kx2)
    Ky = sqrt(Ky2) if Ky2 > 0.0 else 0.0
    K221 = 1.0 + K2 / 2.0

    x = n * (Kx2 - Ky2) / (4.0 * K221)

    Jm = special.jv(nm, x)
    Jp = special.jv(np, x)

    Ax = Kx * (Jp - Jm)
    Ay = Ky * (Jp + Jm)

    fnke = (n / K221) ** 2 * (Ax * Ax + Ay * Ay)
    qnke = K221 * fnke / n
    return fnke, qnke


def fqnk(n: np.ndarray, K: np.ndarray):
    """n is integer order of Fn(K)"""
    nm = (n - 1) // 2
    np_ = (n + 1) // 2

    K2 = K * K
    K221 = 1.0 + K2 / 2.0

    x = np.outer(n, K2) / (4.0 * K221)

    Jm = np.empty(x.shape)
    Jp = np.empty(x.shape)
    for i in range(n.size):
        Jm[i] = special.jv(nm[i], x[i])
        Jp[i] = special.jv(np_[i], x[i])

    tmp = np.outer(n, K) / K221 * (Jm - Jp)
    # breakpoint()
    fnk = tmp * tmp
    qnk = K221 * fnk / n[..., np.newaxis]
    return fnk, qnk


def calc_brill(
    l="?",
    nKvals=101,
    Kmin=0.5,
    Kmax=3.0,
    n=100,
    ebeam=1.722,
    curr=0.1,
    emitx=4.4,
    emity=0.066,
    betx=14.0,
    bety=3.4,
    sige=0.001,
    mode=1,
):

    global Fkn, F, FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Calculated

    if l == "?":
        print(
            "calc_brill(l='?', nKvals=101,Kmin=0.5, Kmax=3., n=100, ebeam=1.722,"
            "curr=0.1, emitx=4.4, emity=0.066, betx=14., bety=3.4,sige=0.001, mode=1)"
        )
        return

    dK = (Kmax - Kmin) / (nKvals - 1)
    Kvals = np.arange(Kmin, Kmax + dK, dK)
    b0 = Kvals / (echarge1 * L / 1000.0 / (2.0 * pi1 * emasskg1 * clight1))

    if type(KyxList) != int:
        print(
            "\n"
            "***********************************************************\n"
            "Attention: For elliptical undulators K is shift-dependend!\n"
            "Thus K must actually set for each harmonic...\n\n"
            "***********************************************************\n\n"
        )
        time.sleep(3)

    Fkn = [None] * 12
    Qn = [None] * 12
    F = [None] * 12
    FB = [None] * 12
    FC = [None] * 12
    FD = [None] * 12
    B = [None] * 12
    Lam = [None] * 12
    Harm = [None] * 12
    Sigr = [None] * 12
    Sigrp = [None] * 12
    i_list = np.arange(12)
    i_list[0] = 1
    if type(KyxList) == int:
        Fkn, Qn = fqnk(i_list, Kvals)
    else:
        Fkn, Qn = fqnke(i_list, Kvals, KyxList)

    sigx = np.sqrt(emitx * 1.0e-9 * betx) * 1000.0  # mm
    sigxp = np.sqrt(emitx * 1.0e-9 / betx) * 1000.0  # mrad
    sigy = np.sqrt(emity * 1.0e-9 * bety) * 1000.0
    sigyp = np.sqrt(emity * 1.0e-9 / bety) * 1000.0

    if mode == 1:
        print(
            "\n"
            "\Mode=1 (Kim):\n"
            "  sigr := sqrt(lambda*length)/4/pi\n"
            "  sigrp := sqrt(lambda/length)\n"
            "  sigr*sigrp := lambda/(4*pi)"
        )
    elif mode == 2:
        print(
            "\n"
            "Mode=2 (Walker, recommended):\n"
            "  sigr := sqrt(2*lambda*length)/2/pi\n"
            "  sigrp := sqrt(lambda/2/length)\n"
            "  sigr*sigrp := lambda/(2*pi)"
        )
    elif mode == -1:
        print(
            "\n"
            "Mode=-1 Erik Wallen, MAX-IV:\n"
            "  sigw[i]=0.36/[i]/[N] | width of harmonic\n"
            "  mulam[i]=sqrt(1.+(2.*[sigE]*[i]*[N]/0.36)**2)\n"
            "  sigr[i]=sqrt(lam1/[i]*[N]*[l]/(8.*pi**2))\n"  # like Kim
            "  sigrp[i]=sqrt(mulam[i]*lam1/[i]/(2.*[N]*[l]))\n"  # rad like Walker
            "  sigr*sigrp := lambda/(4*pi)"  # like Kim
        )
    elif mode == 3:
        print(
            "\n"
            "Mode=3:\n"
            "  sigr := sqrt(sqrt(2)*lambda*length)/2/pi\n"
            "  sigrp := sqrt(lambda/2/sqrt(2)/length)\n"
            "  sigr*sigrp := lambda/sqrt(2)/(2*pi)"
        )
    else:
        print("\n*** Mode has no meaning, set to 1 (Kim) *** \n")
        mode = 1

    time.sleep(1)

    if not Calculated:
        print(
            "\n For the flux calculation the enery-spread is not taken into account,\n"
            " since it has no effect for the brillant flux and a smaller effect for \n"
            " the max. flux compared to the effect on the flux-density.\n"
            " The flux is always calculated according to Kim's formular (xray\n"
            " data booklet eqn. 17). This overestimates the flux by a factor\n"
            " of two, but since the max. is about a this factor higher than\n"
            " the on-resonant flux, this seems to be alright.\n\n"
        )
        Calculated = True

    time.sleep(1)

    pi = pi1

    for i in 1, 3, 5, 7, 9, 11:

        lami = (
            13.056
            * (1.0 + Kvals ** 2 / 2.0)
            * L
            / 10.0
            / ebeam ** 2
            / 1.0e10
            * 1.0e3
            / i
        )  # mm
        Lam[i] = lami

        harmi = i * 0.950 / (1.0 + Kvals ** 2 / 2.0) / (l / 10.0) * ebeam ** 2  # keV
        Harm[i] = harmi

        rsigphi = (1.0 / (i * n * 2.0 * np.sqrt(2.0))) / np.sqrt(
            (1.0 / (i * n * 2.0 * np.sqrt(2.0))) ** 2 + (2.0 * sige) ** 2
        )

        if mode == 1:
            sigrpi = np.sqrt(lami / (n * l))  # rad
            sigri = lami / 4.0 / pi / sigrpi  # mm
        elif mode == 2:
            sigrpi = np.sqrt(lami / 2.0 / (n * l))  # rad
            sigri = lami / 2.0 / pi / sigrpi  # mm
        elif mode == -1:
            # MAX IV formulas (Erik Wallen?)
            sigwi = 0.36 / i / n  # width of harmonic
            mulami = np.sqrt(1.0 + (2.0 * sigE * i * n / 0.36) ** 2)
            rsigphi = 0.0
            sigri = np.sqrt(lam1 / i * n * l / (8.0 * pi ** 2))  # like Kim
            sigrpi = np.sqrt(mulami * lam1 / i / (2.0 * n * l))  # rad # like Walker

        elif mode == 3:
            sigrpi = np.sqrt(lami / (n * l * 2.0 * np.sqrt(2.0)))  # rad
            sigri = np.sqrt(np.sqrt(2.0) * lami * (n * l)) / 2.0 / pi  # mm

        Sigr[i] = sigri
        Sigrp[i] = sigrpi

        sigrpi = sigrpi * 1000  # mrad

        sigrxi = np.sqrt(sigx ** 2 + sigri ** 2)
        sigrpxi = np.sqrt(sigxp ** 2 + sigrpi ** 2)
        sigryi = np.sqrt(sigy ** 2 + sigri ** 2)
        sigrpyi = np.sqrt(sigyp ** 2 + sigrpi ** 2)

        rsigxyi = sigrpi ** 2 / (sigrpxi * sigrpyi)

        F[i] = 1.431e14 * n * Qn[i] * curr  # flux, data booklet Kim
        FD[i] = (
            1.744e14 * n ** 2 * ebeam ** 2 * curr * Fkn[i]
        )  # spectral brightness, data booklet Kim
        FB[i] = FD[i] * 2.0 * pi * sigrpi ** 2  # brilliant flux
        FD[i] = FD[i] * rsigphi * rsigxyi  # d.h. keine Emittanzwirkung auf FB!

        if mode == 1:
            B[i] = (
                F[i] * rsigphi / (2.0 * pi) ** 2 / (sigrxi * sigrpxi * sigryi * sigrpyi)
            )
        elif mode == 2:
            B[i] = (
                FB[i]
                * rsigphi
                / (2.0 * pi) ** 2
                / (sigrxi * sigrpxi * sigryi * sigrpyi)
            )
        elif mode == -1:
            # MAX IV formulas (Erik Wallen?)
            rsigxyi = sigrpi ** 2 / (sigrpxi * sigrpyi)
            # flux, data booklet Kim divived by 2 (MAX IV formula)
            F[i] = 1.431e14 * n * Qn[i] * curr / 2.0
            # flux, data booklet Kim divived by 2 (MAX IV formula)
            FB[i] = 1.431e14 * n * Qn[i] * curr / 2.0
            B[i] = FB[i] / (2.0 * pi) ** 2 / (sigrxi * sigrpxi * sigryi * sigrpyi)
        elif mode == 3:
            B[i] = (
                FB[i]
                * rsigphi
                / (2.0 * pi) ** 2
                / (sigrxi * sigrpxi * sigryi * sigrpyi)
            )

        FC[i] = B[i] * (lami / 2.0) ** 2 * 1.0e6  # mrad**2 -> rad**2!!


def write_brill(fout="brilliance.dat"):
    global Fkn, F, FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    with open(fout, "w") as file:
        file.write(
            "* n, Harm, Flux-dens., Flux, Brilliance, Fbrill, Fcoh, SigR, SigRp\n"
        )
        ll = len(Harm[1]) - 1
        for i in 1, 3, 5, 7, 9, 11:
            for l in range(len(Harm[i])):
                k = ll - l
                template = (f"{i} " + 8 * " {:.4g}" + "\n").format
                file.write(
                    template(
                        Harm[i][k],
                        FD[i][k],
                        F[i][k],
                        B[i][k],
                        FB[i][k],
                        FC[i][k],
                        Sigr[i][k],
                        Sigrp[i][k],
                    )
                )


def _writelastrun():
    _UpdateVars()
    with open(".pybrill_last.dat", "w") as file:
        values = (
            L,
            nKvals,
            Kmin,
            Kmax,
            Kellip,
            Nmin,
            Nmax,
            N,
            Ebeam,
            Curr,
            Emitx,
            Emity,
            Betx,
            Bety,
            Sige,
            Mode,
        )
        file.write("".join(f"{value}\n") for value in values)


def _readlastrun():
    global L, nKvals, Kmin, Kmax, Kellip, Nmin, Nmax, N, Ebeam, Curr, Emitx, Emity, Betx, Bety, Sige, Mode
    try:
        with open(".pybrill_last.dat") as file:
            L = float(file.readline().strip())
            nKvals = int(file.readline().strip())
            Kmin = float(file.readline().strip())
            Kmax = float(file.readline().strip())
            line = file.readline().strip().lower()
            Kellip = line in "true", "1", "yes", "y", "j", "ja"
            Nmin = int(file.readline().strip())
            Nmax = int(file.readline().strip())
            N = int(file.readline().strip())
            Ebeam = float(file.readline().strip())
            Curr = float(file.readline().strip())
            Emitx = float(file.readline().strip())
            Emity = float(file.readline().strip())
            Betx = float(file.readline().strip())
            Bety = float(file.readline().strip())
            Sige = float(file.readline().strip())
            Mode = int(file.readline().strip())
    except FileNotFoundError:
        pass


def pwplot(y, n, ftit):
    global L, nKvals, Kvals, b0, Kmin, Kmax, Kellip, N, Ebeam, Curr, Emitx, Emity
    global Betx, Bety, Sige, Mode, KyzList

    x = Harm[n]
    plt.plot(x, y)

    with open(f"{ftit}_Harm_{n}.dat", "w") as file:
        file.write(f"* Period-length {L}\n")
        file.write(f"* Beam energy [GeV], Curr [A] {Ebeam} {Curr}\n")
        file.write(f"* Hori. and vert. Emittance [nm-rad] {Emitx} {Emity}\n")
        file.write(f"* Hori. and vert. Beta-functions [m] {Betx} {Bety}\n")
        file.write(f"* Rel. beam energy spread {Sige}\n")
        for j in range(nKvals):
            i = nKvals - j - 1
            file.write(f"{j+1} {x[i]} {y[i]} {Kvals[i]} {b0[i]}\n")

        print(f"Written to {file.name}")


def grid(alpha=0.5):
    global Fig, Ax, Grid
    if Grid:
        plt.grid(Grid)
        Ax.grid(which="minor", alpha=alpha)


def zoom(xmin, xmax, ymin, ymax):
    global Fig, Ax
    Ax.axis([xmin, xmax, ymin, ymax])

    grid()
    plt.show(block=False)


def _pcohflux():
    global Fkn, F, FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Nmin, Nmax, Emin, Emax, Bmin, Bmax, Fdmin, Fdmax, Fmin, Fmax, FBmin, FBmax, Fcmin, Fcmax
    global Fig, Ax, Curr

    if Calculated == False:
        _calc()

    Fig = plt.gcf()
    Fig.clear()

    Nmin = max(Nmin, 1)
    Nmax = min(Nmax, 11)

    for i in [1, 3, 5, 7, 9, 11]:
        if i >= Nmin and i <= Nmax:
            pwplot(FC[i], i, "Coh_Flux")

    Fig = plt.gcf()

    Ax = plt.gca()
    Ax.set_xscale("log")
    Ax.set_yscale("log")

    Ax.set_title("Coherent Flux")

    Ax.set_xlabel("photon enery [keV]")
    Ax.set_ylabel("N [1/s/mm$^{2}$/0.1%BW/" + str(Curr * 1000.0) + "mA]")

    if FCmin == -1.0 and FCmax == -1.0:
        ymin, ymax = Ax.get_ylim()
        Ax.set_ylim(ymax / 1.0e3, ymax)

    grid()
    plt.show(block=False)


def _pbrillflux():
    global Fkn, F, FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Nmin, Nmax, Emin, Emax, Bmin, Bmax, Fdmin, Fdmax, Fmin, Fmax, FBmin, FBmax, Fcmin, Fcmax
    global Fig, Ax, Curr

    if Calculated == False:
        _calc()

    Fig = plt.gcf()
    Fig.clear()

    Nmin = max(Nmin, 1)
    Nmax = min(Nmax, 11)

    for i in [1, 3, 5, 7, 9, 11]:
        if i >= Nmin and i <= Nmax:
            pwplot(FB[i], i, "Brilliant_Flux")

    Fig = plt.gcf()

    Ax = plt.gca()
    Ax.set_xscale("log")
    Ax.set_yscale("log")

    Ax.set_title("Brilliant Flux")

    Ax.set_xlabel("photon enery [keV]")
    Ax.set_ylabel("N [1/s/mm$^{2}$/0.1%BW/" + str(Curr * 1000.0) + "mA]")

    if FBmin == -1.0 and FBmax == -1.0:
        ymin, ymax = Ax.get_ylim()
        Ax.set_ylim(ymax / 1.0e4, ymax)

    grid()
    plt.show(block=False)


def _pflux():
    global Fkn, F, FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Nmin, Nmax, Emin, Emax, Bmin, Bmax, Fdmin, Fdmax, Fmin, Fmax, FBmin, FBmax, Fcmin, Fcmax
    global Fig, Ax, Curr

    if Calculated == False:
        _calc()

    Fig = plt.gcf()
    Fig.clear()

    Nmin = max(Nmin, 1)
    Nmax = min(Nmax, 11)

    for i in [1, 3, 5, 7, 9, 11]:
        if i >= Nmin and i <= Nmax:
            pwplot(F[i], i, "Flux")

    Fig = plt.gcf()

    Ax = plt.gca()
    Ax.set_xscale("log")
    Ax.set_yscale("log")

    Ax.set_title("Flux")

    Ax.set_xlabel("photon enery [keV]")
    Ax.set_ylabel("N [1/s/0.1%BW/" + str(Curr * 1000.0) + "mA]")

    if Fmin == -1.0 and Fmax == -1.0:
        ymin, ymax = Ax.get_ylim()
        Ax.set_ylim(ymax / 1.0e4, ymax)

    grid()
    plt.show(block=False)


def _pfluxden():
    global Fkn, F, FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Nmin, Nmax, Emin, Emax, Bmin, Bmax, Fdmin, Fdmax, Fmin, Fmax, FBmin, FBmax, Fcmin, Fcmax
    global Fig, Ax, Curr

    if Calculated == False:
        _calc()

    Fig = plt.gcf()
    Fig.clear()

    Nmin = max(Nmin, 1)
    Nmax = min(Nmax, 11)

    for i in [1, 3, 5, 7, 9, 11]:
        if i >= Nmin and i <= Nmax:
            pwplot(FD[i], i, "Flux-density")

    Fig = plt.gcf()

    Ax = plt.gca()
    Ax.set_xscale("log")
    Ax.set_yscale("log")

    Ax.set_title("Flux-density")

    Ax.set_xlabel("photon enery [keV]")
    Ax.set_ylabel("N [1/s/mrad$^{2}$/0.1%BW/" + str(Curr * 1000.0) + "mA]")

    if FDmin == -1.0 and FDmax == -1.0:
        ymin, ymax = Ax.get_ylim()
        Ax.set_ylim(ymax / 1.0e4, ymax)

    grid()
    plt.show(block=False)


def _pbrill():
    global Fkn, F, FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global Nmin, Nmax, Emin, Emax, Bmin, Bmax, Fdmin, Fdmax, Fmin, Fmax, FBmin, FBmax, Fcmin, Fcmax
    global Fig, Ax, Curr, Vsetup

    if Calculated == False:
        _calc()

    Fig = plt.gcf()
    Fig.clear()

    Nmin = max(Nmin, 1)
    Nmax = min(Nmax, 11)

    for i in [1, 3, 5, 7, 9, 11]:
        if i >= Nmin and i <= Nmax:
            pwplot(B[i], i, "Brilliance")

    Fig = plt.gcf()

    Ax = plt.gca()
    Ax.set_xscale("log")
    Ax.set_yscale("log")

    Ax.set_title("Brilliance")

    Ax.set_xlabel("photon enery [keV]")
    Ax.set_ylabel("N [1/s/mm$^{2}$/mrad$^{2}$/0.1%BW/" + str(Curr * 1000.0) + "mA]")

    if Bmin == -1.0 and Bmax == -1.0:
        ymin, ymax = Ax.get_ylim()
        Ax.set_ylim(ymax / 1.0e4, ymax)

    grid()
    plt.show(block=False)


def _UpdateVars():
    global Vsetup, Nmin, Nmax, Kellip, KyxList
    global L, nKvals, Kvals, b0, Kmin, Kmax, Kellip, N, Ebeam, Curr, Emitx, Emity, Betx, Bety, Sige, Mode

    L = float(Vsetup[0][1])
    nKvals = int(float(Vsetup[1][1]))
    Kmin = float(Vsetup[2][1])
    Kmax = float(Vsetup[3][1])
    Kellip = Vsetup[4][1]
    if type(Kellip) == str:
        if Kellip.lower() in ["true", "1", "yes", "y", "j", "ja"]:
            Kellip = True
        else:
            Kellip = False
    else:
        Kellip = bool(Vsetup[4][1])
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

    if Kellip:
        KyxList = [1.0, 0.42, 0.32, 0.27, 0.24, 0.22]
    else:
        KyxList = 0

    dK = (Kmax - Kmin) / (nKvals - 1)
    Kvals = np.arange(Kmin, Kmax + dK, dK)
    b0 = Kvals / (echarge1 * L / 1000.0 / (2.0 * pi1 * emasskg1 * clight1))


def _calc():
    global Calculated, LastSetup, Kellip
    global L, nKvals, Kvals, b0, Kmin, Kmax, Kellip, N, Ebeam, Curr, Emitx, Emity, Betx, Bety, Sige, Mode

    Calculated = True
    _UpdateVars()

    calc_brill(
        L, nKvals, Kmin, Kmax, N, Ebeam, Curr, Emitx, Emity, Betx, Bety, Sige, Mode
    )

    # if type(LastSetup) == int:
    #    print("\n",Vsetup,"\n")


def _exit():

    try:
        _writelastrun()
    except:
        pass

    if platform.system() == "Windows":
        stat = os.system("taskkill /F /PID " + str(os.getpid()))
    else:
        stat = os.system("kill " + str(os.getpid()))


def _SetUpIn(event, kvar):
    global LastSetup
    LastSetup = [event, kvar]


def _SetUpOut(event, kvar):
    global Vsetup, LastSetup

    ev = LastSetup[0].widget
    val = ev.get()
    Vsetup[kvar][1] = val

    if kvar == 4:
        v = Vsetup[4][1]
        if type(v) == str:
            if v.lower() in ["true", "1", "yes", "y", "j", "ja"]:
                Kellip = True
            else:
                Kellip = False
        else:
            Kellip = bool(Kellip)


def _closeSetUp():
    global Vsetup, LastSetup, Kellip, KyxList

    if LastSetup:
        ev = LastSetup[0].widget
        kvar = LastSetup[1]
        val = ev.get()
        Vsetup[kvar][1] = val

    v = Vsetup[4][1]
    if type(v) == str:
        if v.lower() in ["true", "1", "yes", "y", "j", "ja"]:
            Kellip = True
        else:
            Kellip = False
    else:
        Kellip = bool(Kellip)

    SetUp.destroy()


# def _closeSetUp(win)


def _setup():

    global Fkn, F, FD, FC, FB, Qn, B, Harm, Lam, Sigr, Sigrp, KyxList
    global L, nKvals, Kvals, b0, Kmin, Kmax, Kellip, N, Ebeam, Curr, Emitx, Emity, Betx, Bety, Sige, Mode
    global Mmenu, Myfont
    global SetUp, Vsetup, LastSetup

    SetUp = Toplevel()
    LastSetup = 0

    for i in range(len(Vsetup)):
        f = Frame(SetUp)
        flab = Label(f, text=Vsetup[i][0])
        fent = Entry(f)
        fent.insert(1, Vsetup[i][1])
        flab.pack(side=LEFT)
        fent.pack(side=RIGHT)
        fent.bind("<FocusIn>", lambda event, kvar=i: _SetUpIn(event, kvar))
        fent.bind("<FocusOut>", lambda event, kvar=i: _SetUpOut(event, kvar))
        f.pack(fill="x")

    bClose = Button(SetUp, text="Ok", command=_closeSetUp)
    bClose.pack()

    v = Vsetup[4][1]
    if type(v) == str:
        if v.lower() == ["true", "1", "yes", "ja"]:
            Kellip = True
        else:
            Kellip = False
    else:
        Kellip = bool(Kellip)

    if Kellip:
        print(
            "\n"
            "\n***********************************************************\n"
            "Attention: For elliptical undulators K is shift-dependend!\n"
            "Thus K must actually set for each harmonic...\n"
            "***********************************************************\n"
        )


def getgeo():

    fig = plt.gcf()
    geo = fig.canvas.manager.window.wm_geometry()
    git = geo.split("+")
    wh = git[0].split("x")
    x = int(git[1])
    y = int(git[2])
    w = int(wh[0])
    h = int(wh[1])

    return w, h, x, y


def _showMenu(menu):

    global Mmenu, Omenu, Toolbar, NMmenu, NOmenu, Myfont

    fontsize = int(Myfont[1])
    w, h, x, y = getgeo()

    menu.post(x + int(w * 0.35), y + h - 2 * fontsize * (NMmenu + 1))


def _canbutbrill(ev):
    global Mmenu, Omenu, NMmenu, NOmenu
    print("_canbutbrill", ev)


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
print(" sigr[i]=sqrt(lam1/[i]*[N]*[l]/(8.*pi**2))")  # like Kim
print(" sigrp[i]=sqrt(mulam[i]*lam1/[i]/(2.*[N]*[l]))")  # rad like Walker
print(" sigr*sigrp := lambda/(4*pi)\n\n")  # like Kim

print("\n Mode=3 (WAVE):")
print("---------------\n")
print("\n sigr := sqrt(sqrt(2)*lambda*length)/2/pi")
print(" sigrp := sqrt(lambda/2/sqrt(2)/length)")
print(" sigr*sigrp := lambda/sqrt(2)/(2*pi)\n\n")

print("-----------------------------------------------------------------------\n")

print("\n For the flux calculation the enery-spread is not taken into account, since")
print(" it has no effect for the brillant flux and a smaller effect for the max. flux")
print(" compared to the effect on the flux-density.")
print(" The flux is always calculated according to Kim's formular (xray")
print(" data booklet eqn. 17). This overestimates the flux by a factor")
print(" of two, but since the max. is about a this factor higher than")
print(" the on-resonant flux, this seems to be alright.\n\n")

print("-----------------------------------------------------------------------\n")

L = 50.0
nKvals = 101
Kmin = 0.5
Kmax = 3.0
Kellip = False
Nmin = 1
Nmax = 11
N = 100
Ebeam = 1.722
Curr = 0.1
Emitx = 4.4
Emity = 0.066
Betx = 14.0
Bety = 3.4
Sige = 0.001
Mode = 2  # Walker

_readlastrun()

Vsetup = [
    ["Period-length [mm]", L],
    ["Number of K values", nKvals],
    ["Kmin", Kmin],
    ["Kmax", Kmax],
    ["Elliptical Undulator", Kellip],
    ["Lowest harmonic", Nmin],
    ["Highest harmonic", Nmax],
    ["Number of periods", N],
    ["Beam energy [GeV]", Ebeam],
    ["Current [A]", Curr],
    ["Hor. Emit. [nm-rad]", Emitx],
    ["Ver. Emit. [nm-rad]", Emity],
    ["Hori. Beta function", Betx],
    ["Vert. Beta function", Bety],
    ["Rel. energy spread", Sige],
    ["Mode [-1,1,2,3]", Mode],
]

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

Emin = -1.0
Emax = -1.0

Bmin = -1.0
Bmax = -1.0
Fmin = -1.0
Fmax = -1.0
FDmin = -1.0
FDmax = -1.0
FCmin = -1.0
FCmax = -1.0
FBmin = -1.0
FBmax = -1.0

KyxList = 0

Grid = True
LastSetup = 0

mpl.use("TkAgg")

Fig = plt.figure()
Fig.show()

# Ax = plt.gca()
# Fig.canvas.mpl_disconnect(CanButId)

Wmain = plt.gcf()
Wmaster = Wmain.canvas.toolbar.master

Toolbar = Wmain.canvas.toolbar
Myfont = ("arial", 13)

CanButBrill = Wmain.canvas.mpl_connect("button_press_event", _canbutbrill)

Mmenu = Menu(Toolbar, tearoff=1, font=Myfont)
mPlot = Menu(Mmenu, tearoff=1, font=Myfont)

##########
NMmenu += 1
Mmenu.add_command(label="Set Up", command=_setup)

NMmenu += 1
Mmenu.add_command(label="Calculate", command=_calc)

NMmenu += 1
Mmenu.add_cascade(label="Plot", menu=mPlot)

NMmenu += 1
Mmenu.add_command(label="Exit", command=_exit)
##########

mPlot.add_command(label="Brilliance", command=_pbrill)
mPlot.add_command(label="Flux-density", command=_pfluxden)
mPlot.add_command(label="Flux", command=_pflux)
mPlot.add_command(label="Coherent Flux", command=_pcohflux)
mPlot.add_command(label="Brilliant Flux", command=_pbrillflux)

##########

bMmenu = Button(
    Toolbar, text="Menu", font=Myfont, command=lambda menu=Mmenu: _showMenu(menu)
)

bMmenu.pack(side=LEFT)


def main():
    _setup()
    _pbrill()


if __name__ == "__main__":
    main()
