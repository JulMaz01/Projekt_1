from math import sin, cos, tan, sqrt, asin, atan, atan2, degrees, radians, pi
import numpy as np

class Transformacje:
    def __init__(self, model: str = "wgs84"):       
        """
        PARAMETRY ELIPSOIDY:
            a - duża półoś elipsoidy - promień równikowy
            b - mała półoś elipsoidy - promień południkowy
            flat - spłaszczenie
            ecc2 - mimośród^2
 
        + WGS84: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
        + Inne powierzchnie odniesienia: https://en.wikibooks.org/wiki/PROJ.4#Spheroid
        + Parametry planet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html
      
        """   
        if model == "wgs84":
            self.a = 6378137.0 # semimajor_axis
            self.b = 6356752.31424518 # semiminor_axis
        elif model == "grs80":
            self.a = 6378137.0
            self.b = 6356752.31414036
        elif model == "mars":
            self.a = 3396900.0
            self.b = 3376097.80585952
        else:
            raise NotImplementedError(f"{model} model not implemented")
        self.flat = (self.a - self.b) / self.a
        self.ecc = sqrt(2 * self.flat - self.flat ** 2) # eccentricity  WGS84:0.0818191910428 
        self.ecc2 = (2 * self.flat - self.flat ** 2) # eccentricity**2

# ---------------------------------------------------------------------------------------------

    def func_n(self, f: float) -> float:
        """
        Funkcja obliczająca promień krzywizny w I wertykale
 
        Parameters
        ----------                
        INPUT:
            phi : [float] - szerokość geodezyjna (decimal degrees)

        Returns
        -------     
        OUTPUT:
            N : [float] - promien krzywizny (meters)
        """
        N = (self.a)/sqrt(1-self.ecc2*(sin(f)**2))
        
        return(N)

# ---------------------------------------------------------------------------------------------
    
    def xyz2plh(self, X, Y, Z, output = 'dec_degree'):
        """
        Algorytm Hirvonena - algorytm transformacji współrzędnych ortokartezjańskich (X, Y, Z)
        na współrzędne geodezyjne: długość, szerokość i wysokość elipsoidalna (phi, lam, h). Jest to proces iteracyjny. 
        W wyniku 3-4-krotneej iteracji wyznaczenia wsp. phi można przeliczyć współrzędne z dokładnoscią ok 1 cm.     
        
        Parameters
        ----------        
        INPUT: Współrzędne w układzie ortokartezjańskim
            X : [float] - współrzędna geocentryczna (meters)
            Y : [float] - współrzędna geocentryczna (meters)
            Z : [float] - współrzędna geocentryczna (meters)
 
        Returns
        -------        
        OUTPUT:  
            phi : [float] - szerokość geodezyjna (decimal degrees)
            lam : [float] - długość geodezyjna (decimal degrees)
            h   : [float] - wysokość elipsoidalna (meters)

         OUTPUT OPTIONS: 
            dec_degree - decimal degree
            dms - degree, minutes, sec
            
        """
        r   = sqrt(X**2 + Y**2)           # promień
        lat_prev = atan(Z / (r * (1 - self.ecc2)))    # pierwsze przyblizenie
        lat = 0
        while abs(lat_prev - lat) > 0.000001/206265:    
            lat_prev = lat
            N = self.a / sqrt(1 - self.ecc2 * sin(lat_prev)**2)
            h = r / cos(lat_prev) - N
            lat = atan((Z/r) * (((1 - self.ecc2 * N/(N + h))**(-1))))
        lon = atan(Y/X)
        N = self.a / sqrt(1 - self.ecc2 * (sin(lat))**2);
        h = r / cos(lat) - N       
        if output == "dec_degree":
            return degrees(lat), degrees(lon), h 
        elif output == "dms":
            lat = self.deg2dms(degrees(lat))
            lon = self.deg2dms(degrees(lon))
            return f"{lat[0]:02d}:{lat[1]:02d}:{lat[2]:.2f}", f"{lon[0]:02d}:{lon[1]:02d}:{lon[2]:.2f}", f"{h:.3f}"
        else:
            raise NotImplementedError(f"{output} - output format not defined")
            

# ---------------------------------------------------------------------------------------------

    def flh2xyz(self, f:float, l:float, h:float):
        """
        Algorytm transformacji współrzędnych geodezyjnych (phi, lam, h) na współrzędne ortokartezjańskie (X, Y, Z)

        Parameters
        ----------        
        INPUT:
            phi : [float] - szerokość geodezyjna (decimal degrees)
            lam : [float] - długość geodezyjna (decimal degrees)
            h   : [float] - wysokość elipsoidalna (meters)
            
        Returns
        -------
        OUTPUT: Współrzędne w układzie ortokartezjańskim
            X : [float] - współrzędna geocentryczna (meters)
            Y : [float] - współrzędna geocentryczna (meters)
            Z : [float] - współrzędna geocentryczna (meters) 

        """
        N = self.func_n(f)
        X = (N + h)*cos(f)*cos(l)
        Y = (N + h)*cos(f)*sin(l)
        X = (N*(1-self.ecc2)+h)*sin(f)
        
        return(X,Y,Z)

 # ---------------------------------------------------------------------------------------------               

    def neu(self, f, l, h, X, Y, Z):
        """
        Algorytm obliczający współrzędne geodezyjne wektora przestrzennego neu. 
    
        Parameters
        ----------
        INPUT:        
            phi : [float] - szerokość geodezyjna (decimal degrees)
            lam : [float] - długość geodezyjna (decimal degrees)
            h   : [float] - wysokość elipsoidalna (meters)
            X : [float] - współrzędna geocentryczna (meters)
            Y : [float] - współrzędna geocentryczna (meters)
            Z : [float] - współrzędna geocentryczna (meters)
    
        Returns
        -------        
        OUTPUT:
            n,e,u : [float] - współrzędne geodezyjne wektora przestrzennego
    
        """   
        N = self.func_n(f)
        
        Xp = (N + h) * cos(f) * cos(l)
        Yp = (N + h) * cos(f) * sin(l)
        Zp = (N * (1 - self.ecc2) + h) * sin(f)
        
        XYZp = np.array([Xp, Yp, Zp])
        XYZs = np.array([X, Y, Z])
        
        XYZ = XYZs - XYZp
        XYZ = np.array(XYZ)
        	
        Rneu = np.array([[-sin(f)*cos(l), -sin(l), cos(f)*cos(l)],
                         [-sin(f)*sin(l), cos(l), cos(f)*sin(l)],
                         [cos(f), 0, sin(f)]])
        
        n, e, u = Rneu.T @ XYZ
      
        return(n, e, u)
 
 # ---------------------------------------------------------------------------------------------
    
    def sigma (self, f: float) -> float:        
        """
        Algorytm obliczący długosć łuku południka, który przyjmuje f wyliczone z hirvonena

        Parameters
        ----------
        INPUT:
            f : [float] - szerokość geodezyjna (decimal degrees)

        Returns
        -------            
        OUTPUT:
            si : [float] - długosć łuku południka (meters)
            
        """               
        A0 = 1 - (self.ecc2/4)-((3*(self.ecc2**2))/64)-((5*(self.ecc2**3))/256)
        A2 = (3/8)*(self.ecc2+((self.ecc2**2)/4)+((15*(self.ecc2**3))/128))
        A4 = (15/256)*((self.ecc2**2)+((3*(self.ecc2**3))/4))
        A6 = (35*(self.ecc2**3))/3072
        si = self.a*(A0*f - A2*sin(2*f) + A4*sin(4*f) - A6*sin(6*f))

        return(si)

 # ---------------------------------------------------------------------------------------------
    
    def get_l0(self, l: float) -> int:
         
        """
        Algorytm obliczający wartosć południka srodkowego L0 na podstawie l
        
        INPUT:
            l : [float] - długość geodezyjna (decimal degrees)
            
        OUTPUT:
            L0 : [float] - południk srodkowy w danym układzie (radians)
        """        
        if 13.5 < l <= 16.5:
            L0 = 15 * pi / 180
        if 16.5 < l <= 19.5:
            L0 = 18 * pi / 180
        if 19.5 < l <= 22.5:
            L0 = 21 * pi / 180
        if 22.5 < l <= 25.5:
            L0 = 24 * pi / 180
            
        return(L0)
    
 # ---------------------------------------------------------------------------------------------   
    
    def fl2xy(self, phi: float, lam: float) -> tuple:
        """
        Algorytm przeliczający współrzędne godezyjne (phi, lam) na współrzędne w 
        odwzorowaniu Gaussa-Krugera (xgk, ygk)

        Parameters
        ----------
        INPUT:
            phi : [float] - szerokość geodezyjna (decimal degrees)
            lam : [float] - długość geodezyjna (decimal degrees)

        Returns
        -------
        OUTPUT:
            xgk :[float] - współrzędna x w odwzorowaniu Gaussa-Krugera (meters)
            ygk :[float] - współrzędna y w odwzorowaniu Gaussa-Krugera (meters)

        """
        f = phi*pi/180
        l = lam*pi/180
        b2 = (self.a**2)*(1-self.ecc2)
        ep2 = (self.a**2-b2)/b2
        t = tan(f)
        n2 = ep2*(cos(f)**2)
        N = self.func_n(f)
        si = self.sigma(f)
        dL = l - self.get_l0(lam) 
        xgk = si + ((dL**2)/2)*N*sin(f)*cos(f)*(1 + (dL**2/12)*cos(f)**2*(5 - t**2 + 9*n2 + 4*n2**2) + (dL**4/360)*cos(f)**4*(61 - 58*t**2 + t**4 + 270*n2 - 330*n2*t**2))
        ygk = dL*N*cos(f)*(1 + (dL**2/6)*cos(f)**2*(1 - t**2 + n2) + (dL**4/120)*cos(f)**4*(5 - 18*t**2 + t**4 + 14*n2 - 58*n2*t**2))
       
        return(xgk,ygk)

# ---------------------------------------------------------------------------------------------

    def u2000(self, f:float, l:float):
        """
        Algorytm przeliczający współrzędne geodezyjne (phi, lam) na współrzędne w układzie PL-2000.

        Parameters
        ----------            
        INPUT:
            f : [float] - szerokość geodezyjna (decimal degrees)
            l : [float] - długość geodezyjna (decimal degreees)

        Returns
        -------    
        OUTPUT:
            x00 : [float] - współrzędna X w układzie PL-2000
            y00 : [float] - współrzędna Y w układzie PL-2000
    
        """       
        m2000 = 0.999923
        xgk, ygk = self.fl2xy(f, l)
        l0 = self.get_l0(l)
        x00 = xgk * m2000
        y00 = ygk * m2000 + (l0*180/pi/3)* 1000000 + 500000;
    
        return (x00, y00)

# ---------------------------------------------------------------------------------------------

    def u1992(self, f:float, l:float):      
        """
        Algorytm przeliczający współrzędne geodezyjne (phi, lam) na współrzędne w układzie 1992.

        Parameters
        ----------      
        INPUT:
            f : [float] - szerokość geodezyjna (decimal degreees)
            l : [float] - długość geodezyjna (decimal degreees)
    
        Returns
        -------    
        OUTPUT:
            x92 : [float] - współrzędna X w układzie PL-1992
            y92 : [float] - współrzędna Y w układzie PL-1992
    
        """
        xgk, ygk = self.fl2xy(f, l)
        x92 = xgk * 0.9993-5300000
        y92 = ygk *0.9993+500000
        
        return(x92,y92)

# ---------------------------------------------------------------------------------------------

    def azymut_elewacja(self, f, l, h, X, Y, Z):
        """
        Algorytm obliczający elewację (kąt horyzontalny) i azymut na podstawie współrzędnych geodezyjnych wektora przestrzennego neu. 
    
        Parameters
        ----------
        INPUT:        
            phi : [float] - szerokość geodezyjna (decimal degrees)
            lam : [float] - długość geodezyjna (decimal degrees)
            h   : [float] - wysokość elipsoidalna (meters)
            X : [float] - współrzędna geocentryczna (meters)
            Y : [float] - współrzędna geocentryczna (meters)
            Z : [float] - współrzędna geocentryczna (meters)
    
        Returns
        -------        
        OUTPUT:
            el : [float] - kąt horyzontalny (decimal degree)
            Az : [float] - azymut (decimal degree)
    
        """   
        n, e, u = self.neu(f, l, h, X, Y, Z)

        azymut = atan2(e, n)
        azymut = np.rad2deg(azymut)
        azymut = azymut + 360 if azymut < 0 else azymut

        elewacja = asin(u/(sqrt(e**2+n**2+u**2)))
        elewacja = np.rad2deg(elewacja)

        return(azymut, elewacja)

# ---------------------------------------------------------------------------------------------

    def odl2D(self, A, B):
        """
        Algorytm obliczający odległosć pomiędzy dwoma punktami A, B o współrzędnych (X,Y)  
        
       Parameters
       ----------      
       INPUT: Współrzędne w układzie ortokartezjańskim
           X : [float] - współrzędna geocentryczna (meters)
           Y : [float] - współrzędna geocentryczna (meters)
        
        Returns
        -------    
        OUTPUT:
            odl_2D : [float] - odległość pomiędzy punktami A, B (meters)
    
        """
        x1, y1 = A[0], A[1]
        x2, y2 = B[0], B[1]
        odl_2D = sqrt((x2 - x1)**2+(y2 - y1)**2)

        return(odl_2D)

# ---------------------------------------------------------------------------------------------

    def odl3D(self, A, B):
        """
        Algorytm obliczający odległosć pomiędzy dwoma punktami A, B o współrzędnych przestrzennych-3D (X,Y,Z)  
        
       Parameters
       ----------      
       INPUT: Współrzędne w układzie ortokartezjańskim
           X : [float] - współrzędna geocentryczna (meters)
           Y : [float] - współrzędna geocentryczna (meters)
           Z : [float] - współrzędna geocentryczna (meters)
        
        Returns
        -------    
        OUTPUT:
            odl_3D : [float] - odległość 3D pomiędzy punktami A, B (meters)
    
        """
        x1, y1, z1 = A[0], A[1], A[2]
        x2, y2, z2 = B[0], B[1], B[2]
        odl_3D = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
 
        return(odl_3D)

# ---------------------------------------------------------------------------------------------

    naglowek = 'Transormacja współrzędnych geodezyjnych \\ Julia Mazurkiewicz \\ 01160141@pw.edu.pl\n\n  X               Y               Z                 f              l              h             n                e                u             x00             y00            x92            y92               Az             el              2D             3D\n'

# ---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # utworzenie obiektu
    geo = Transformacje(model = "wgs84")
    # dane XYZ geocentryczne
    X = 3664940.500; Y = 1409153.590; Z = 5009571.170
    phi, lam, h = geo.xyz2plh(X, Y, Z)
    print(phi, lam, h)
