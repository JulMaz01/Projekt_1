from math import sin, cos, tan, sqrt, asin, atan, atan2, degrees, radians
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
    
    def xyz2flh(self, X, Y, Z, output = 'dec_degree'):
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
            phi : [float] : szerokość geodezyjna (decimal degrees)
            lam : [float] : długość geodezyjna (decimal degrees)
            h   : [float] : wysokość elipsoidalna (meters)

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

    def flh2XYZ(self, f, l, h):
        """
        Algorytm transformacji współrzędnych geodezyjnych (phi, lam, h) na współrzędne ortokartezjańskie (X, Y, Z)

        Parameters
        ----------        
        INPUT:
            phi : [float] : szerokość geodezyjna (radians)
            lam : [float] : długość geodezyjna (radians)
            h   : [float] : wysokość elipsoidalna (meters)
            
        Returns
        -------
                OUTPUT: Współrzędne w układzie ortokartezjańskim
            X : [float] - współrzędna geocentryczna (meters)
            Y : [float] - współrzędna geocentryczna (meters)
            Z : [float] - współrzędna geocentryczna (meters) 

        """
        N = self.a / sqrt(1 - self.ecc2 * (sin(l))**2);
        X = (N+h)*cos(f)*cos(l)
        Y = (N+h)*cos(f)*sin(l)
        Z = (N*(1-self.ecc2)+h)*sin(f)
        
        return(X,Y,Z)
 
 # ---------------------------------------------------------------------------------------------       
 
    def get_mean_of_x_y_z(coordinates: np.ndarray):          
        """
        Algorytm obliczający srednią arytmetyczną współrzędnych (X,Y,Z)
        
        Parameters
        ----------
        INPUT:        
            [X, Y, Z] : [float] - tablica współrzędnych geocentrycznych (meters)
  
        Returns
        -------        
        OUTPUT:
            X_mean : [float] -  współrzędna referencyjna X (meters)
            Y_mean : [float] -  współrzędna referencyjna Y (meters)
            Z_mean : [float] -  współrzędna referencyjna Z (meters)   
        
        """            
        x_mean, y_mean, z_mean = coordinates.mean(axis=0)
        return x_mean, y_mean, z_mean

 # ---------------------------------------------------------------------------------------------               

    def enu(self, X, Y, Z, X_mean, Y_mean, Z_mean):
        """
        Algorytm obliczający współrzędne geodezyjne wektora przestrzennego ENU. 
    
        Parameters
        ----------
        INPUT:        
            X_mean : [float] -  współrzędna referencyjna X (meters)
            Y_mean : [float] -  współrzędna referencyjna Y (meters)
            Z_mean : [float] -  współrzędna referencyjna Z (meters)
            X : [float] - współrzędna geocentryczna (meters)
            Y : [float] - współrzędna geocentryczna (meters)
            Z : [float] - współrzędna geocentryczna (meters)
    
        Returns
        -------        
        OUTPUT:
            ENU : [float] - lista złożona z 3-elementów: E, N, U
    
        """   
        f, l, h = self.xyz2flh(X, Y, Z)
    
        delta_X = X - X_mean
        delta_Y = Y - Y_mean
        delta_Z = Z - Z_mean
    
        Rt = np.matrix([((-sin(f) * cos(l)), (-sin(f) * sin(l)), (cos(f))),
                        ((-sin(l)), (cos(l)), (0)),
                        ((cos(f) * cos(l)), (cos(f) * sin(l)), (sin(f))),])
    
        d = np.matrix([delta_X, delta_Y, delta_Z])
        d = d.T
        neu = Rt * d
        enu = neu[1], neu[0], neu[2]
        
        return enu
 
    
 # ---------------------------------------------------------------------------------------------
    
    def sigma(self, f):        
        """
        Algorytm obliczący długosć łuku południka.

        Parameters
        ----------
        INPUT:
            f  :[float] : szerokość geodezyjna (radians)

        Returns
        -------            
        OUTPUT:
            si :[float] : długosć łuku południka (meters)
            
        """               
        A0 = 1 - (self.ecc2 / 4) - (3 / 64) * (self.ecc2**2) - (5 / 256) * (self.ecc2**3)
        A2 = (3 / 8) * (self.ecc2 + (self.ecc2**2) / 4 + (15 / 128) * (self.ecc2**3))
        A4 = (15 / 256) * (self.ecc2**2 + 3 / 4 * (self.ecc2**3))
        A6 = (35 / 3072) * self.ecc2**3
        si = self.a * (A0 * f - A2 * sin(2 * f) + A4 * sin(4 * f) - A6 * sin(6 * f))

        return si
    
 # ---------------------------------------------------------------------------------------------   
    
    def fl2xy(self, f, l, L0):
        """
        Algorytm przeliczający współrzędne godezyjne (phi, lam) na współrzędne w 
        odwzorowaniu Gaussa-Krugera (xgk, ygk)

        Parameters
        ----------
        INPUT:
            phi : [float] - szerokość geodezyjna (radians)
            lam : [float] - długość geodezyjna (radians)
            L0  : [float] - południk srodkowy w danym układzie (radians)

        Returns
        -------
        OUTPUT:
            xgk :[float] : współrzędna x w odwzorowaniu Gaussa-Krugera (meters)
            ygk :[float] : współrzędna y w odwzorowaniu Gaussa-Krugera (meters)

        """
        b2 = (self.a**2) * (1 - self.ecc2)
        ep2 = (self.a**2 - b2) / b2
        t = tan(f)
        n2 = ep2 * (cos(f) ** 2)
        N = self.a / sqrt(1 - self.ecc2 * (sin(l))**2)
        si = self.sigma(f)
        dL = l - L0
        xgk = si + (dL**2 / 2) * N * sin(f) * cos(f) * (1 + (dL**2 / 12) * cos(f) ** 2 * (5 - t**2 + 9 * n2 + 4 * n2**2) + (dL**4 / 360) * cos(f) ** 4 * (61 - 58 * t**2 + t**4 + 270 * n2 - 330 * n2 * t**2))
        ygk = (dL * N * cos(f) * ( 1 + (dL**2 / 6) * cos(f) ** 2 * (1 - t**2 + n2) + (dL**4 / 120) * cos(f) ** 4 * (5 - 18 * t**2 + t**4 + 14 * n2 - 58 * n2 * t**2)))

        return (xgk, ygk)

# ---------------------------------------------------------------------------------------------

    def u2000(self, f, l):
        """
        Algorytm przeliczający współrzędne geodezyjne (phi, lam) na współrzędne w układzie PL-2000.

        Parameters
        ----------            
        INPUT:
            f   :[float] : szerokość geodezyjna (radiany)
            l   :[float] : długość geodezyjna (radiany)

        Returns
        -------    
        OUTPUT:
            x00 :[float] : współrzędna X w układzie 2000
            y00 :[float] : współrzędna Y w układzie 2000
    
        """
        L0 = (np.floor((f + 1.5) / 3)) * 3   
        xgk, ygk = self.fl2xy(f, l, L0)    
        m2000 = 0.999923    
        x00 = xgk * m2000
        y00 = ygk * m2000 + L0 / 3 * 1000000 + 500000
    
        return (x00, y00)

# ---------------------------------------------------------------------------------------------

    def u1992(self, xgk, ygk, L0):      
        """
        Algorytm przeliczający współrzędne geodezyjne (phi, lam) na współrzędne w układzie 1992.

        Parameters
        ----------      
        INPUT:
            f   :[float] : szerokość geodezyjna (radiany)
            l   :[float] : długość geodezyjna (radiany)
    
        Returns
        -------    
        OUTPUT:
            x92 :[float] : współrzędna X w układzie 1992
            y92 :[float] : współrzędna Y w układzie 1992
    
        """        
        m92 = 0.9993        
        x92 = xgk * m92 - 5300000
        y92 = ygk * m92 + 500000
        
        return(x92,y92)

# ---------------------------------------------------------------------------------------------

    def odl2D(A, B):
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
        odl_2D = sqrt( (A[0] - B[0])**2 + (A[1] - B[1])**2 )

        return(odl_2D)

# ---------------------------------------------------------------------------------------------

    def odl3D(A, B):
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
        odl_3D = sqrt( (A[0] - B[0])**2 + (A[1] - B[1])**2 + (A[2] - B[2])**2 )
 
        return(odl_3D)

# ---------------------------------------------------------------------------------------------

    def Azymut(self, enu):
        """
        Algorytm obliczający azymut na podstawie współrzędnych geodezyjnych wektora przestrzennego ENU.
        
       Parameters
       ----------      
       INPUT:
           ENU : [float] - lista złożona z 3-elementów: E, N, U
        
        Returns
        -------    
        OUTPUT:
            Az : [float] - azymut (decimal degree)
    
        """       
        Az = atan2(enu[0],enu[1])
        Az = np.rad2deg(Az)
        Az = Az + 360 if Az < 0 else Az
        
        return(Az)

# ---------------------------------------------------------------------------------------------

    def Elewacja(self, enu):
        """
        Algorytm obliczający elewację (kąt horyzontalny) na podstawie współrzędnych geodezyjnych wektora przestrzennego ENU.
        
       Parameters
       ----------      
       INPUT:
           ENU : [float] - lista złożona z 3-elementów: E, N, U
        
        Returns
        -------    
        OUTPUT:
           el : [float] - kąt horyzontalny (decimal degree)
    
        """
        el = asin(enu[2]/(sqrt(enu[0]**2+enu[1]**2+enu[2]**2)))
        el = np.rad2deg(el)
        
        return(el)
 
# ---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # utworzenie obiektu
    geo = Transformacje(model = "wgs84")
    # dane XYZ geocentryczne
    X = 3664940.500; Y = 1409153.590; Z = 5009571.170
    phi, lam, h = geo.xyz2plh(X, Y, Z)
    print(phi, lam, h)
    phi, lam, h = geo.xyz2plh2(X, Y, Z)
    print(phi, lam, h)
        
    
