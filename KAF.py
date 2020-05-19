class QKLMS:
    def __init__(self, eta=None, epsilon=None, sigma=None):
        if eta == None:
          self.eta = 0.9 #Remplazar por algun criterio
        else:
          self.eta = eta
        if epsilon == None:
          self.epsilon = 10 #Remplazar por algun criterio
        else:
          self.epsilon = epsilon
        self.sigma = sigma
        self.CB = [] #Codebook
        self.a_coef = [] #Coeficientes
        self.__CB_cov = [] #Covarianzas
        self.__CB_cov_sums = [] #Sumas acumuladas
        self.__CB_cov_prods = [] #Productos acumulados
        self.__n_cov = [] # n iteraciones en covarianza
        self.CB_growth = [] #Crecimiento del codebook por iteracion 
        
    def evaluate(self, u , d):
        import numpy as np
        if u.shape[0]!=d.shape[0]:
          raise ValueError('All of the input arguments must be of the same lenght')
        if self.sigma == None:
            from scipy.spatial.distance import cdist
       	    d_sgm = cdist(u,u)
       	    self.sigma = np.median(d_sgm) #Criterio de la mediana      
        #Tamaños
        N,D = u.shape
        Nd,Dd = d.shape
        #Inicializaciones
        self.CB.append(u[0,:]) #Codebook
        self.a_coef.append(self.eta*d[0,:]) #Coeficientes
        self.__CB_cov.append(np.eye(D)) #Covarianzas
        self.__CB_cov_sums.append(np.zeros((D,)))
        self.__CB_cov_prods.append(np.zeros((D,D)))
        self.__n_cov.append(1)
        #Salida
        y = np.empty((Nd,Dd))
        i = 1
        # err = 0.1

        while True:
          yi,disti = self.__output(u[i,:].reshape(-1,D)) #Salida
          d_mahal = self.__dmahal(u[i,:].reshape(-1,D)) #Distancia de Mahalanobis 
          # self.__newEta(yi,err) #Nuevo eta
          err = d[i] - yi # Error
          #Cuantizacion
          min_index = np.argmin(d_mahal)  
          if disti[min_index] <= self.epsilon:
            self.a_coef[min_index] =(self.a_coef[min_index] + self.eta*err).item()
            self.__CB_cov[min_index] = self.__naiveCovQ(min_index,u[i,:])
            self.__n_cov[min_index] = self.__n_cov[min_index] + 1
          else:
            self.CB.append(u[i,:])
            self.a_coef.append((self.eta*err).item())
            self.__CB_cov.append(np.eye(D))
            self.__CB_cov_sums.append(np.zeros((D,)))
            self.__CB_cov_prods.append(np.zeros((D,D)))
            self.__n_cov.append(1)
            
          self.CB_growth.append(len(self.CB)) #Crecimiento del diccionario 
          y[i-1] = yi
          i+=1      
          if(i == N-1):
            return y

    def __output(self,ui):
        from scipy.spatial.distance import cdist
        import numpy as np
        dist = cdist(np.asarray(self.CB), ui)
        K = np.exp(-0.5*(dist**2)/(self.sigma**2))
        y = K.T.dot(np.asarray(self.a_coef))
        return [y,dist]

    def __newEta(self, y, errp):
        # y: Salida calculada
        # errp: Error a priori 
        self.eta = (2*errp*y)/(errp**2 + 1)
        return False

    def __dmahal(self,ui):
        import numpy as np
        from scipy.spatial import distance
        i = 0
        dist_m = np.array([])
        while i < len(self.CB):    
          # VI = np.linalg.inv(np.trace(np.cov(np.asarray(self.CB)[i], ui))*np.eye(ui.shape[1]))
          d_i = distance.mahalanobis(np.asarray(self.CB)[i],ui,self.__CB_cov[i]/self.__n_cov[i])
          dist_m = np.concatenate((dist_m,[d_i]))
          i+=1
        return dist_m

    def __naiveCovQ(self,index_cov,ui):
        import numpy as np
        sums = np.asarray(self.__CB_cov_sums[index_cov])
        prods = np.asarray(self.__CB_cov_prods[index_cov])
        n = self.__n_cov[index_cov]
        sums = np.sum(np.concatenate((sums.reshape(-1,1),ui.reshape(-1,1)), axis=1),axis=1)
        means = np.outer(sums,sums)/n
        prods += np.outer(ui,ui)
        cov = prods - means
        self.__CB_cov_sums[index_cov] = sums
        self.__CB_cov_prods[index_cov] = prods
        return cov