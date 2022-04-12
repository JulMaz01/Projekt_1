#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from transformacje import *

geo = Transformacje(model = input('WYBIERZ UKŁAD --> wgs84, grs80, mars: '))

plik = "wsp_inp.txt"
# odczyt z pliku: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.genfromtxt.html
tablica = np.genfromtxt(plik, delimiter=',', skip_header = 4)

# ------------------------------------- XYZ to plh -------------------------------------------------------

plh = []
for i in range(tablica.shape[0]):
    plh.append(geo.xyz2plh(tablica[i][0], tablica[i][1], tablica[i][2]))

tablica = np.c_[tablica, np.array(plh)]

# ------------------------------------- XYZ, plh to neu -------------------------------------------------------

neu = []
for i in range(tablica.shape[0]):
    neu.append(geo.neu(tablica[i][3], tablica[i][4], tablica[i][5], tablica[i][0], tablica[i][1], tablica[i][2]))

tablica = np.c_[tablica, np.array(neu)]

# ------------------------------------- pl to x00 y00 -------------------------------------------------------

xy_2000 = []
for x in range(tablica.shape[0]):
    xy_2000.append(geo.u2000(tablica[i][3], tablica[i][4]))

tablica = np.c_[tablica, np.array(xy_2000)]

# ------------------------------------- pl to x92 y92 -------------------------------------------------------

xy_1992 = []
for x in range(tablica.shape[0]):
    xy_1992.append(geo.u1992(tablica[i][3], tablica[i][4]))

tablica = np.c_[tablica, np.array(xy_1992)]

# ------------------------------------- XYZ, plh to Az, el -------------------------------------------------------

az_el = []
for x in range(tablica.shape[0]):
    az_el.append(geo.azymut_elewacja(tablica[i][3], tablica[i][4], tablica[i][5], tablica[i][0], tablica[i][1], tablica[i][2]))

tablica = np.c_[tablica, np.array(az_el)]

# ------------------------------------- A, B to odl2d, odl3D -------------------------------------------------------

odl_2D_3D = []
for i in range(tablica.shape[0]):
    try:
        odl_2D_3D.append([geo.odl2D(tablica[i], tablica[i+1]), geo.odl3D(tablica[i], tablica[i+1])])
    except IndexError:
        odl_2D_3D.append([0,0])

tablica = np.c_[tablica, np.array(odl_2D_3D)]



#zapis: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.savetxt.html
np.savetxt("wsp_out.txt", tablica, delimiter=',    ', fmt = '%10.3f', header = geo.naglowek)