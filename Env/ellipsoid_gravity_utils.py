# This code translated Matlab code with header given below:
# function [X,Y,Z,LAM]=Ellipsoid_Gravity_Field(x,a,b,c,rho)
# Ellipsoid_Gravity_Field: Determines the components of attraction of a solid
# homogeneous ellipsoid of constant density with semiaxes a > b > c at an
# external point. The solution is derived from the Ivory's method as given
# in MacMillan, 1958 p.60.
#
#    INPUT:
#          x: particle position in body fixed coordinates.
#       a b c: are the semiaxes of the ellipsoid a>b>c to avoid singularity
#
#   OUTPUT:
#     Gx,Gy,Gz: are the components of attraction in body fixed
#               coordinates.
#          LAM: positive root of the polynomial.
#
#   RELATED FUNCTIONS:
#         lellipf.m; lellipe.m; rf.m; rd.m
#
# Dario Cersosimo, October 16, 2009.
#                  Feb 25, 2010. Added LAM to the outputs.





import numpy as np

def ellipsoid_gravity_field(x, a, b , c, rho):
    
    G = 6.6695e-11                     # Gravitational constant [kg m**3/s**2]
    aa = a**2
    bb = b**2
    cc = c**2
    errtol = 1e-10

    # find the greatest root LAM
    B = -(x[0]**2 + x[1]**2 + x[2]**2 - aa - bb - cc)
    C = -cc*(x[0]**2 + x[1]**2) + bb*(cc - x[0]**2 - x[2]**2) + aa*(bb +cc - x[1]**2 - x[2]**2)
    D = -bb*cc*x[0]**2 + aa*(-cc*x[1]**2 + bb*(c - x[2])*(c + x[2]))
    poly = np.asarray([1, B, C, D])
    LAM = np.max(np.roots(poly))
    # find elliptic integrals F1 and E1
    phi = np.arcsin(np.sqrt((aa-cc)/(LAM + aa)))# Argument phi, s.t. 0 < phi <= np.pi/2
    k = np.sqrt((aa-bb)/(aa-cc))            # Modulus k, s.t. 0 < k < 1
    F1 = lellipf(phi, k, errtol)
    E1 = lellipe(phi, k, errtol)
    fac1 = 4*np.pi*G*rho*a*b*c/np.sqrt(aa-cc)
    fac2 = np.sqrt((aa-cc)/((aa+LAM)*(bb+LAM)*(cc+LAM)))

    # Components of attraction.
    X = fac1/(aa-bb)*(E1-F1)
    Y = fac1*((-aa+cc)*E1/((aa-bb)*(bb-cc)) + F1/(aa-bb)+ (cc+LAM)*fac2/(bb-cc))

    Z = fac1*(-E1 + (bb+LAM)*fac2)/(cc-bb)
    return np.asarray([X,Y,Z]), LAM


#
# lellipe(phi, k, errtol)
#
# Inputs:
#
#   phi     Input angle vector size 1xN.
#   k       Input parameter vector size 1 or 1xN.
#   errtol  Error tolerance for Carlson's algorithms.
#
# Matlab function to compute Legendre's (incomplete) elliptic integral
# E(phi, k).  Uses a vectorized implementation of Carlson's Duplication Algorithms
# for symmetric elliptic integrals as found in "Computing Elliptic
# Integrals by Duplication," by B. C. Carlson, Numer. Math. 33, 1-16 (1979)
# and also found in ACM TOMS Algorithm 577.  Section 4 in the paper cited
# here describes how to convert between the symmetric elliptic integrals
# and Legendre's elliptic integrals.
#
# Returns NaN's for any argument values outside input range.
#

def lellipe(phi, k, errtol):

    # Argument checking for vectorization:

    phivec = phi
    kvec = k

    snphi = np.sin(phivec)
    csphi = np.cos(phivec)
    snphi2 = snphi**2
    csphi2 = csphi**2
    k2 = kvec**2
    y = 1.0 - k2*snphi2
    onesvec = 1 
    f = snphi * rf(csphi2,  y, onesvec, errtol) - k2 * snphi * snphi2 * rd(csphi2, y, onesvec, errtol)/3.0

    return f

#
# lellipf(phi, k, errtol)
#
# Inputs:
#
#   phi     Input angle vector size 1 or 1xN.
#   k       Input parameter vector size 1 or 1xN.
#   errtol  Error tolerance for Carlson's algorithms.
#
# Matlab function to compute Legendre's (incomplete) elliptic integral
# F(phi, k).  Uses a vectorized implementation of Carlson's Duplication Algorithms
# for symmetric elliptic integrals as found in "Computing Elliptic
# Integrals by Duplication," by B. C. Carlson, Numer. Math. 33, 1-16 (1979)
# and also found in ACM TOMS Algorithm 577.  Section 4 in the paper cited
# here describes how to convert between the symmetric elliptic integrals
# and Legendre's elliptic integrals.
#
# Returns NaN's for any argument values outside input range.
#

def lellipf(phi, k, errtol):

    phivec = phi 
    kvec = k 

    snphi = np.sin(phivec)
    csphi = np.cos(phivec)
    csphi2 = csphi * csphi
    onesvec = 1 
    y = onesvec - kvec*kvec * snphi*snphi
    f = snphi * rf(csphi2,  y, onesvec, errtol)
    return f



# Elliptic Integrals by Duplication," by B. C. Carlson, Numer. Math.
# 33, 1-16 (1979).
#
# Returns NaN's for any argument values outside input range.
#
# Algorithm is also from Carlson's ACM TOMS Algorithm 577.
#
# This code is a complete rewrite of the algorithm in vectorized form.
# It was not produced by running a FORTRAN to Matlab converter.
#
# The following text is copied from ACM TOMS Algorithm 577 FORTRAN code:
#
#   X AND Y ARE THE VARIABLES IN THE INTEGRAL RC(X,Y).
#
#   ERRTOL IS SET TO THE DESIRED ERROR TOLERANCE.
#   RELATIVE ERROR DUE TO TRUNCATION IS LESS THAN
#   16 * ERRTOL ** 6 / (1 - 2 * ERRTOL).
#
#   SAMPLE CHOICES:  ERRTOL     RELATIVE TRUNCATION
#                               ERROR LESS THAN
#                    1.D-3      3.D-19
#                    3.D-3      2.D-16
#                    1.D-2      3.D-13
#                    3.D-2      2.D-10
#                    1.D-1      3.D-7
#
# Note by TRH:
#
#   Absolute truncation error when the integrals are order 1 quantities
#   is closer to errtol, so be careful if you want high absolute precision.
#
# Thomas R. Hoffend Jr., Ph.D.
# 3M Company
# 3M Center Bldg. 236-GC-26
# St. Paul, MN 55144
# trhoffendjr@mmm.com
#

def rf(x, y, z, errtol):
    realmin = 1e-100
    realmax = 1e100
    # Argument limits as set by Carlson:
    LoLim = 5.0 * realmin
    UpLim = 5.0 * realmax


    # Define internally acceptable variable ranges for iterations:
    Xi = x
    Yi = y
    Zi = z

    # Carlson's duplication algorithm for Rf:
    Xn = Xi
    Yn = Yi
    Zn = Zi
    Mu = (Xn + Yn + Zn) / 3.0
    Xndev = 2.0 - (Mu + Xn) / Mu
    Yndev = 2.0 - (Mu + Yn) / Mu
    Zndev = 2.0 - (Mu + Zn) / Mu
    epslon = np.max( np.abs([Xndev, Yndev, Zndev]) )
    while epslon >= errtol:
        Xnroot = np.sqrt(Xn)
        Ynroot = np.sqrt(Yn)
        Znroot = np.sqrt(Zn)
        lambda1 = Xnroot * (Ynroot + Znroot) + Ynroot * Znroot
        Xn = 0.25 * (Xn + lambda1)
        Yn = 0.25 * (Yn + lambda1)
        Zn = 0.25 * (Zn + lambda1)
        Mu = (Xn + Yn + Zn) / 3.0
        Xndev = 2.0 - (Mu + Xn) / Mu
        Yndev = 2.0 - (Mu + Yn) / Mu
        Zndev = 2.0 - (Mu + Zn) / Mu
        epslon = np.max( np.abs([Xndev , Yndev , Zndev]) )

    C1 = 1.0 / 24.0
    C2 = 3.0 / 44.0
    C3 = 1.0 / 14.0
    E2 = Xndev * Yndev - Zndev * Zndev
    E3 = Xndev * Yndev * Zndev
    S = 1.0 + (C1 * E2 - 0.1 - C2 * E3) * E2 + C3 * E3
    f = S / np.sqrt(Mu)

    # Return NaN's where input argument was out of range:
    return f

#
# rd(x, y, z, errtol)
#
# Inputs:
#
#   x       Input vector size 1xN.
#   y       Input vector size 1xN.
#   z       Input vector size 1xN.
#   errtol  Error tolerance.
#
# Matlab function to compute Carlson's symmetric elliptic integral Rd.
# Implementation of Carlson's Duplication Algorithm 4 in "Computing
# Elliptic Integrals by Duplication," by B. C. Carlson, Numer. Math.
# 33, 1-16 (1979).
#
# Returns NaN's for any argument values outside input range.
#
# Algorithm is also from Carlson's ACM TOMS Algorithm 577.
#
# This code is a complete rewrite of the algorithm in vectorized form.
# It was not produced by running a FORTRAN to Matlab converter.
#
# The following text is copied from ACM TOMS Algorithm 577 FORTRAN code:
#
#   X AND Y ARE THE VARIABLES IN THE INTEGRAL RC(X,Y).
#
#   ERRTOL IS SET TO THE DESIRED ERROR TOLERANCE.
#   RELATIVE ERROR DUE TO TRUNCATION IS LESS THAN
#   16 * ERRTOL ** 6 / (1 - 2 * ERRTOL).
#
#   SAMPLE CHOICES:  ERRTOL     RELATIVE TRUNCATION
#                               ERROR LESS THAN
#                    1.D-3      3.D-19
#                    3.D-3      2.D-16
#                    1.D-2      3.D-13
#                    3.D-2      2.D-10
#                    1.D-1      3.D-7
#
# Note by TRH:
#
#   Absolute truncation error when the integrals are order 1 quantities
#   is closer to errtol, so be careful if you want high absolute precision.
#
# Thomas R. Hoffend Jr., Ph.D.
# 3M Company
# 3M Center Bldg. 236-GC-26
# St. Paul, MN 55144
# trhoffendjr@mmm.com
#

def rd(x, y, z, errtol ):

    realmin = 1e-100
    realmax = 1e100

    # Argument limits as set by Carlson:
    LoLim = 5.0 * realmin
    UpLim = 5.0 * realmax


    # Define internally acceptable variable ranges for iterations:
    Xi = x
    Yi = y
    Zi = z

    # Carlson's duplication algorithm for Rf:
    Xn = Xi
    Yn = Yi
    Zn = Zi
    sigma = 0.0
    power4 = 1.0

    Mu = (Xn + Yn + 3.0 * Zn) * 0.2
    Xndev = (Mu - Xn) / Mu
    Yndev = (Mu - Yn) / Mu
    Zndev = (Mu - Zn) / Mu
    epslon = np.max( np.abs([Xndev, Yndev, Zndev]) )
    while epslon >= errtol:
        Xnroot = np.sqrt(Xn)
        Ynroot = np.sqrt(Yn)
        Znroot = np.sqrt(Zn)
        lambda1 = Xnroot * (Ynroot + Znroot) + Ynroot * Znroot
        sigma = sigma + power4 / (Znroot * (Zn + lambda1))
        power4 = 0.25 * power4
        Xn = 0.25 * (Xn + lambda1)
        Yn = 0.25 * (Yn + lambda1)
        Zn = 0.25 * (Zn + lambda1)
        Mu = (Xn + Yn + 3.0 * Zn) * 0.2
        Xndev = (Mu - Xn) / Mu
        Yndev = (Mu - Yn) / Mu
        Zndev = (Mu - Zn) / Mu
        epslon = np.max( np.abs([Xndev, Yndev, Zndev]) )

    C1 = 3.0 / 14.0
    C2 = 1.0 / 6.0
    C3 = 9.0 / 22.0
    C4 = 3.0 / 26.0
    EA = Xndev * Yndev
    EB = Zndev * Zndev
    EC = EA - EB
    ED = EA - 6.0 * EB
    EF = ED + EC + EC
    S1 = ED * (-C1 + 0.25 * C3 * ED - 1.50 * C4 * Zndev * EF)
    S2 = Zndev * (C2 * EF + Zndev * (-C3 * EC + Zndev * C4 * EA))
    f = 3.0 * sigma + power4 * (1.0 + S1 + S2) / (Mu * np.sqrt(Mu))

    return f

