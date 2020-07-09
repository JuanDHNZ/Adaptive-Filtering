class QKLMS2:
    """Filtro QKLMS que aplica distacia de Mahalanobis en la cuantización"""
    def __init__(self, eta=0.9, epsilon=10, sigma=None):
        self.eta = eta #Remplazar por algun criterio
        self.epsilon = epsilon
        self.sigma = sigma
        self.CB = [] #Codebook
        self.a_coef = [] #Coeficientes
        self.__CB_cov = [] #Covarianzas
        self.__CB_cov_sums = [] #Sumas acumuladas
        self.__CB_cov_prods = [] #Productos acumulados
        self.__n_cov = [] # n iteraciones en covarianza
        self.CB_growth = [] #Crecimiento del codebook por iteracion
        self.initialize = True #Bandera de inicializacion
        self.init_eval = True
        self.evals = 0  #
        
        self.fitFlag = False
        
        self.testCB_means = [] #Prueba
        self.testDists = []
        
    ''' 
    Parametros:
        u -> Señal de entrada
        d -> Salida deseada
        
    El metodo evalua cada muestra de entrada, y determina si se debe crear un nuevo centroide
    o si se debe cuantizar de acuerdo al umbral establecido
    '''
    def evaluate(self, u , d):
        import numpy as np
        #Validación d tamaños de entrada
        if len(u.shape) == 2:
            if u.shape[0]!=d.shape[0]:
                raise ValueError('All of the input arguments must be of the same lenght')
        else:
            if len(u.shape) == 1:
                u = u.reshape(1,-1)
                d = d.reshape(1,-1)
        #Sigma definido por criterio de mediana
        if self.sigma == None:
            from scipy.spatial.distance import cdist
       	    d_sgm = cdist(u,u)
       	    self.sigma = np.median(d_sgm) #Criterio de la mediana      
        #Tamaños u y d
        N,D = u.shape
        Nd,Dd = d.shape
        #Inicializaciones
        if self.initialize:
            y = np.empty((Nd-1,Dd))
            self.CB.append(u[0,:]) #Codebook
            self.a_coef.append(self.eta*d[0,:]) #Coeficientes
            self.__CB_cov.append(np.eye(D)) #Covarianzas
            self.__CB_cov_sums.append(np.zeros((D,)))
            self.__CB_cov_prods.append(np.zeros((D,D)))
            self.__n_cov.append(1)
            self.testCB_means.append(np.zeros(2,))#Prueba
            #Salida           
            i = 1
            self.initialize = False
            # err = 0.1
            if u.shape[0] == 1:                
                return 0
        else:
            y = np.empty((Nd,Dd))
            i = 0
        while True:
            yi,disti = self.__output(u[i,:].reshape(-1,D)) #Salida       
            d_mahal = self.__dmahal(u[i,:].reshape(-1,D)) #Distancia de Mahalanobis 
            self.testDists.append(d_mahal)
            # self.__newEta(yi,err) #Nuevo eta
            err = d[i] - yi # Error
            #Cuantizacion
            min_index = np.argmin(d_mahal)
            
            if d_mahal[min_index] <= self.epsilon:
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
              
              self.testCB_means.append(np.zeros(2,)) #Prueba
            
            self.CB_growth.append(len(self.CB)) #Crecimiento del diccionario 
            
            if self.init_eval: 
                y[i-1] = yi
            else:
                y[i] = yi

            if(i == N-1):
                # print(len(self.__CB_cov))
                # for k in range(len(self.__CB_cov)):
                #     print(self.__n_cov[k])
                #     print(self.__CB_cov[k]/self.__n_cov[k])
                if self.init_eval:
                    self.init_eval = False           
                return y
            i+=1 

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
        #List comprehension
        #np.asarray(self.CB)[i]
        dist_m = [distance.mahalanobis(self.__CB_cov_sums[k]/self.__n_cov[k],ui,self.__CB_cov[k]/self.__n_cov[k]) for k in range(len(self.CB))]
        dist_m = np.array(dist_m)
        
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
        self.testCB_means[index_cov] = sums/n 
        return cov
    
    def fit(self, u=None, d=None):
        import numpy as np
        #Validaciones
        if u is None:
            raise ValueError("Parameter u is missing")
        if d is None:
            raise ValueError("Parameter d is missing")
        if len(u) != len(d):
            raise ValueError("All input arguments must be the same lenght, u shape is {0} and d shape is {1}".format(u.shape, d.shape))
        
        #Sigma definido por criterio de mediana
        if self.sigma == None:
            from scipy.spatial.distance import cdist
       	    d_sgm = cdist(u,u)
       	    self.sigma = np.median(d_sgm) #Criterio de la mediana      
        #Tamaños u y d
        N,D = u.shape
        Nd,Dd = d.shape
        
        if not self.fitFlag:
            self.fitFlag = True
        
        #Inicializacion
        if self.initialize:
            self.CB.append(u[0,:]) #Codebook
            self.a_coef.append(self.eta*d[0,:]) #Coeficientes
            self.__CB_cov.append(np.eye(D)) #Covarianzas
            self.__CB_cov_sums.append(np.zeros((D,)))
            self.__CB_cov_prods.append(np.zeros((D,D)))
            self.__n_cov.append(1)
            self.testCB_means.append(np.zeros(2,))#Prueba
            #Salida           
            i = 1
            self.initialize = False
            # err = 0.1
            if u.shape[0] == 1:                
                raise ValueError("Invalid shape of inpunt argument, must be vector or matrix")
        else: 
            i = 0
            
        while True:
            yi,disti = self.__output(u[i,:].reshape(-1,D)) #Salida       
            d_mahal = self.__dmahal(u[i,:].reshape(-1,D)) #Distancia de Mahalanobis 
            self.testDists.append(d_mahal)
            # self.__newEta(yi,err) #Nuevo eta
            err = d[i] - yi # Error
            #Cuantizacion
            min_index = np.argmin(d_mahal)
            
            if d_mahal[min_index] <= self.epsilon:
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
              
              self.testCB_means.append(np.zeros(2,)) #Prueba
            
            self.CB_growth.append(len(self.CB)) #Crecimiento del diccionario 
            
            if(i == N-1):
                # print(len(self.__CB_cov))
                # for k in range(len(self.__CB_cov)):
                #     print(self.__n_cov[k])
                #     print(self.__CB_cov[k]/self.__n_cov[k])
                if self.init_eval:
                    self.init_eval = False  
                break
            
            i+=1
            
    def predict(self, u , d):
        import numpy as np
        #Validaciones
        if not self.fitFlag:
            raise ValueError("Fit method must be runed first")
        if u is None:
            raise ValueError("Parameter u is missing")
        if d is None:
            raise ValueError("Parameter d is missing")
        if len(u) != len(d):
            raise ValueError("All input arguments must be the same lenght, u shape is {0} and d shape is {1}".format(u.shape, d.shape))
                
        #Tamaños u y d
        N,D = u.shape
        Nd,Dd = d.shape
        y = np.empty((Nd,Dd), dtype=float)
        i = 0
        
        while True:           
            y[i],disti = self.__output(u[i,:].reshape(-1,D)) #Salida       
            
            if(i == N-1):
                if self.init_eval:
                    self.init_eval = False           
                return y
            i+=1
    
    def score(self, y_target , y_pred):
        #Validaciones
        if y_target is None:
            raise ValueError("Parameter y_target is missing")
        if y_pred is None:
            raise ValueError("Parameter y_pred is missing")
        if len(y_target) != len(y_pred):
            raise ValueError("All input arguments must be the same lenght, u shape is {0} and d shape is {1}".format(y_target.shape, y_pred.shape))
                
        from sklearn.metrics import r2_score
        return r2_score(y_target,y_pred)
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"epsilon": self.epsilon, "sigma": self.sigma}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        


class QKLMS:
    """Filtro QKLMS"""
    def __init__(self, eta=0.9, epsilon=10, sigma=None):
        self.eta = eta #Remplazar por algun criterio
        self.epsilon = epsilon
        self.sigma = sigma
        self.CB = [] #Codebook
        self.a_coef = [] #Coeficientes
        self.__CB_cov = [] #Covarianzas
        self.__CB_cov_sums = [] #Sumas acumuladas
        self.__CB_cov_prods = [] #Productos acumulados
        self.__n_cov = [] # n iteraciones en covarianza
        self.CB_growth = [] #Crecimiento del codebook por iteracion
        self.initialize = True #Bandera de inicializacion
        self.init_eval = True
        self.evals = 0  #
        
        self.testCB_means = [] #Prueba
        self.testDists = []
        
    def evaluate(self, u , d):
        import numpy as np
        #Validación d tamaños de entrada
        if len(u.shape) == 2:
            if u.shape[0]!=d.shape[0]:
                raise ValueError('All of the input arguments must be of the same lenght')
        else:
            if len(u.shape) == 1:
                u = u.reshape(1,-1)
                d = d.reshape(1,-1)
        #Sigma definido por criterio de mediana
        if self.sigma == None:
            from scipy.spatial.distance import cdist
       	    d_sgm = cdist(u,u)
       	    self.sigma = np.median(d_sgm) #Criterio de la mediana      
        #Tamaños u y d
        N,D = u.shape
        Nd,Dd = d.shape
        #Inicializaciones
        if self.initialize:
            y = np.empty((Nd-1,Dd))
            self.CB.append(u[0,:]) #Codebook
            self.a_coef.append(self.eta*d[0,:]) #Coeficientes
            #Salida           
            i = 1
            self.initialize = False
            # err = 0.1
            if u.shape[0] == 1:                
                return 0
        else:
            i = 0
            y = np.empty((Nd,Dd))
        while True:
            yi,disti = self.__output(u[i,:].reshape(-1,D)) #Salida       
            # self.__newEta(yi,err) #Nuevo eta
            err = d[i] - yi # Error
            #Cuantizacion
            min_index = np.argmin(disti)
            
            if disti[min_index] <= self.epsilon:
              self.a_coef[min_index] =(self.a_coef[min_index] + self.eta*err).item()
            else:
              self.CB.append(u[i,:])
              self.a_coef.append((self.eta*err).item())
            
            self.CB_growth.append(len(self.CB)) #Crecimiento del diccionario 
            
            if self.init_eval: 
                y[i-1] = yi
            else:
                y[i] = yi

            if(i == N-1):
                if self.init_eval:
                    self.init_eval = False           
                return y
            i+=1 

    def __output(self,ui):
        from scipy.spatial.distance import cdist
        import numpy as np
        dist = cdist(np.asarray(self.CB), ui)
        K = np.exp(-0.5*(dist**2)/(self.sigma**2))
        y = K .T.dot(np.asarray(self.a_coef))
        return [y,dist]

    def __newEta(self, y, errp):
        # y: Salida calculada
        # errp: Error a priori 
        self.eta = (2*errp*y)/(errp**2 + 1)
        return False

