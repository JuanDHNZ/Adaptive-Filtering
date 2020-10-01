class BGMM_KLMS:
    """Filtro QKLMS que aplica distacia de Mahalanobis en la cuantización y en el kernel"""
    def __init__(self, clusters = 1, wcp = 1, eta = 0.9, max_iter = 500, warm_start=False):
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.clusters = clusters
        self.wcp = wcp
        self.eta = eta #Tasa de aprendizaje
        self.CB = [] #Codebook
        self.a_coef = [] #Coeficientes
        self.CB_cov = [] #Covarianzas
        self.CB_means = []#Medias
        self.__CB_cov_sums = [] #Sumas acumuladas
        self.__CB_cov_prods = [] #Productos acumulados
        self.__n_cov = [] # n iteraciones en covarianza
        self.CB_growth = [] #Crecimiento del codebook por iteracion
        self.initialize = True #Bandera de inicializacion
        self.init_eval = True
        self.evals = 0  #
        
        self.fitFlag = False
        
        self.testCB_means = [] #Prueba
        self.testDists = [] #Prueba
        
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
            self.CB_cov.append(np.eye(D)) #Covarianzas
            self.__CB_cov_sums.append(np.zeros((D,)))
            self.__CB_cov_prods.append(np.zeros((D,D)))
            self.__n_cov.append(1)
            self.testCB_means.append(np.zeros(2,))#Prueba
            #Salida           
            i = 1
            self.initialize = False
            # err = 0.1
            if u.shape[0] == 1:                
                return
        else:
            y = np.empty((Nd,Dd))
            i = 0
        dist_mahal_test = []
        while True:
            yi,disti = self.__output(u[i,:].reshape(-1,D)) #Salida 
            d_mahal = self.__dmahal(u[i,:].reshape(-1,D)) #Distancia de Mahalanobis 
            self.testDists.append(d_mahal)
            dist_mahal_test.append(disti) ###TEST
            # self.__newEta(yi,err) #Nuevo eta
            err = d[i] - yi # Error
            #Cuantizacion
            min_index = np.argmin(d_mahal)
            
            if d_mahal[min_index] <= self.epsilon:
              self.a_coef[min_index] =(self.a_coef[min_index] + self.eta*err).item()
              self.CB_cov[min_index] = self.__naiveCovQ(min_index,u[i,:])
              self.__n_cov[min_index] = self.__n_cov[min_index] + 1
            else:
              self.CB.append(u[i,:])
              self.a_coef.append((self.eta*err).item())
              self.CB_cov.append(np.eye(D))
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
        # dist = cdist(np.asarray(self.CB), ui)
        dist = self.__dmahal2(ui)
        # print("Dist shape = ", dist.shape)
        K = np.exp(-0.5*(dist**2)) #Quitar sigma  y dist con mhalanobis
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
        dist_m = [distance.mahalanobis(self.__CB_cov_sums[k]/self.__n_cov[k],ui,self.CB_cov[k]/self.__n_cov[k]) for k in range(len(self.CB))]
        dist_m = np.array(dist_m)
        return dist_m

    def __dmahal2(self,ui):
        import numpy as np
        from scipy.spatial import distance
        #List comprehension
        #np.asarray(self.CB)[i]
        dist_m = [distance.mahalanobis(self.CB[k],ui,self.CB_cov[k]) for k in range(len(self.CB))]
        dist_m = np.array(dist_m)       
        return dist_m.reshape(-1,1)

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
        #Validaciones
        if u is None:
            raise ValueError("Parameter u is missing")
        if d is None:
            raise ValueError("Parameter d is missing")
        if len(u) != len(d):
            raise ValueError("All input arguments must be the same lenght, u shape is {0} and d shape is {1}".format(u.shape, d.shape))
                         
        if not self.fitFlag:
            self.fitFlag = True  
        
        N,D = u.shape
        
        #GMM fit
        from sklearn.mixture import BayesianGaussianMixture as BGMM
        import numpy as np
        bgmm = BGMM(n_components=self.clusters, weight_concentration_prior=self.wcp, max_iter=self.max_iter, warm_start=self.warm_start).fit(u)
        self.bgmm = bgmm
        from scipy.spatial.distance import cdist
        F = [cdist(u, self.bgmm.means_[c].reshape(1,-1), 'mahalanobis', VI=self.bgmm.precisions_[c]) for c in range(self.bgmm.n_components)]
        F = [np.exp((-f**2)/2) for f in F]
        phi = np.concatenate(F,axis=1)
        
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(phi,d)
        self.reg = reg

           
    def predict(self, u):
        import numpy as np
        #Validaciones
        if not self.fitFlag:
            raise ValueError("Fit method must be runed first")
        if u is None:
            raise ValueError("Parameter u is missing")
                    
        #Tamaños u y d
        N,D = u.shape
        y = np.empty((N,), dtype=float)
        i = 0      
        from scipy.spatial.distance import cdist
        F = [cdist(u, self.bgmm.means_[c].reshape(1,-1), 'mahalanobis', VI=self.bgmm.precisions_[c]) for c in range(self.bgmm.n_components)]
        F = [np.exp((-f**2)/2) for f in F]
        phi = np.concatenate(F,axis=1)
        return self.reg.predict(phi)
        
        # y,d = self.__output(u)
        # return y
        while True:           
            y[i],disti = self.__output(u[i,:].reshape(-1,D)) #Salida                  
            if(i == N-1):
                if self.init_eval:
                    self.init_eval = False           
                return y
            i+=1
    
    def score(self, X=None, y=None):
        #Validaciones
        if X is None:
            raise ValueError("Parameter y_true is missing")
        if y is None:
            raise ValueError("Parameter y_pred is missing")
        if len(X) != len(y):
            raise ValueError("All input arguments must be the same lenght, X shape is {0} and y shape is {1}".format(X.shape, y.shape))
                
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        # print("Shapes: \n y:",y.shape," and y_pred:",y_pred.shape)
        return r2_score(y,y_pred)
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"wcp":self.wcp,"clusters":self.clusters}# 

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    # def kernelMatrix(self, u):
    #     u=0

class GMM_KLMS:
    """Filtro QKLMS que aplica distacia de Mahalanobis en la cuantización y en el kernel"""
    def __init__(self, clusters = None, eta = 0.9):
        self.clusters = clusters
        self.eta = eta #Tasa de aprendizaje
        self.CB = [] #Codebook
        self.a_coef = [] #Coeficientes
        self.CB_cov = [] #Covarianzas
        self.CB_means = []#Medias
        self.__CB_cov_sums = [] #Sumas acumuladas
        self.__CB_cov_prods = [] #Productos acumulados
        self.__n_cov = [] # n iteraciones en covarianza
        self.CB_growth = [] #Crecimiento del codebook por iteracion
        self.initialize = True #Bandera de inicializacion
        self.init_eval = True
        self.evals = 0  #
        
        self.fitFlag = False
        
        self.testCB_means = [] #Prueba
        self.testDists = [] #Prueba
        
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
            self.CB_cov.append(np.eye(D)) #Covarianzas
            self.__CB_cov_sums.append(np.zeros((D,)))
            self.__CB_cov_prods.append(np.zeros((D,D)))
            self.__n_cov.append(1)
            self.testCB_means.append(np.zeros(2,))#Prueba
            #Salida           
            i = 1
            self.initialize = False
            # err = 0.1
            if u.shape[0] == 1:                
                return
        else:
            y = np.empty((Nd,Dd))
            i = 0
        dist_mahal_test = []
        while True:
            yi,disti = self.__output(u[i,:].reshape(-1,D)) #Salida 
            d_mahal = self.__dmahal(u[i,:].reshape(-1,D)) #Distancia de Mahalanobis 
            self.testDists.append(d_mahal)
            dist_mahal_test.append(disti) ###TEST
            # self.__newEta(yi,err) #Nuevo eta
            err = d[i] - yi # Error
            #Cuantizacion
            min_index = np.argmin(d_mahal)
            
            if d_mahal[min_index] <= self.epsilon:
              self.a_coef[min_index] =(self.a_coef[min_index] + self.eta*err).item()
              self.CB_cov[min_index] = self.__naiveCovQ(min_index,u[i,:])
              self.__n_cov[min_index] = self.__n_cov[min_index] + 1
            else:
              self.CB.append(u[i,:])
              self.a_coef.append((self.eta*err).item())
              self.CB_cov.append(np.eye(D))
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
        # dist = cdist(np.asarray(self.CB), ui)
        dist = self.__dmahal2(ui)
        # print("Dist shape = ", dist.shape)
        K = np.exp(-0.5*(dist**2)) #Quitar sigma  y dist con mhalanobis
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
        dist_m = [distance.mahalanobis(self.__CB_cov_sums[k]/self.__n_cov[k],ui,self.CB_cov[k]/self.__n_cov[k]) for k in range(len(self.CB))]
        dist_m = np.array(dist_m)
        return dist_m

    def __dmahal2(self,ui):
        import numpy as np
        from scipy.spatial import distance
        #List comprehension
        #np.asarray(self.CB)[i]
        dist_m = [distance.mahalanobis(self.CB[k],ui,self.CB_cov[k]) for k in range(len(self.CB))]
        dist_m = np.array(dist_m)       
        return dist_m.reshape(-1,1)

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
        #Validaciones
        if u is None:
            raise ValueError("Parameter u is missing")
        if d is None:
            raise ValueError("Parameter d is missing")
        if len(u) != len(d):
            raise ValueError("All input arguments must be the same lenght, u shape is {0} and d shape is {1}".format(u.shape, d.shape))
                         
        if not self.fitFlag:
            self.fitFlag = True  
        
        N,D = u.shape
        
        #GMM fit
        from sklearn.mixture import GaussianMixture as GMM
        import numpy as np
        gmm = GMM(n_components=self.clusters).fit(u)
        self.gmm = gmm
        from scipy.spatial.distance import cdist
        F = [cdist(u, self.gmm.means_[c].reshape(1,-1), 'mahalanobis', VI=self.gmm.precisions_[c]) for c in range(self.gmm.n_components)]
        F = [np.exp((-f**2)/2) for f in F]
        phi = np.concatenate(F,axis=1)
        
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(phi,d)
        self.reg = reg

           
    def predict(self, u):
        import numpy as np
        #Validaciones
        if not self.fitFlag:
            raise ValueError("Fit method must be runed first")
        if u is None:
            raise ValueError("Parameter u is missing")
                    
        #Tamaños u y d
        N,D = u.shape
        y = np.empty((N,), dtype=float)
        i = 0      
        from scipy.spatial.distance import cdist
        F = [cdist(u, self.gmm.means_[c].reshape(1,-1), 'mahalanobis', VI=self.gmm.precisions_[c]) for c in range(self.gmm.n_components)]
        F = [np.exp((-f**2)/2) for f in F]
        phi = np.concatenate(F,axis=1)
        return self.reg.predict(phi)
        
        # y,d = self.__output(u)
        # return y
        while True:           
            y[i],disti = self.__output(u[i,:].reshape(-1,D)) #Salida                  
            if(i == N-1):
                if self.init_eval:
                    self.init_eval = False           
                return y
            i+=1
    
    def score(self, X=None, y=None):
        #Validaciones
        if X is None:
            raise ValueError("Parameter y_true is missing")
        if y is None:
            raise ValueError("Parameter y_pred is missing")
        if len(X) != len(y):
            raise ValueError("All input arguments must be the same lenght, X shape is {0} and y shape is {1}".format(X.shape, y.shape))
                
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        # print("Shapes: \n y:",y.shape," and y_pred:",y_pred.shape)
        return r2_score(y,y_pred)
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"clusters": self.clusters}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    # def kernelMatrix(self, u):
    #     u=0


class QKLMS3:
    """Filtro QKLMS que aplica distacia de Mahalanobis en la cuantización y en el kernel"""
    def __init__(self, eta=0.9, epsilon=10, sigma=None):
        self.eta = eta #Remplazar por algun criterio
        self.epsilon = epsilon
        self.sigma = sigma
        self.CB = [] #Codebook
        self.a_coef = [] #Coeficientes
        self.CB_cov = [] #Covarianzas
        self.CB_means = []#Medias
        self.__CB_cov_sums = [] #Sumas acumuladas
        self.__CB_cov_prods = [] #Productos acumulados
        self.__n_cov = [] # n iteraciones en covarianza
        self.CB_growth = [] #Crecimiento del codebook por iteracion
        self.initialize = True #Bandera de inicializacion
        self.init_eval = True
        self.evals = 0  #
        
        self.fitFlag = False
        
        self.testCB_means = [] #Prueba
        self.testDists = [] #Prueba
        
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
            self.CB_cov.append(np.eye(D)) #Covarianzas
            self.__CB_cov_sums.append(np.zeros((D,)))
            self.__CB_cov_prods.append(np.zeros((D,D)))
            self.__n_cov.append(1)
            self.testCB_means.append(np.zeros(2,))#Prueba
            #Salida           
            i = 1
            self.initialize = False
            # err = 0.1
            if u.shape[0] == 1:                
                return
        else:
            y = np.empty((Nd,Dd))
            i = 0
        dist_mahal_test = []
        while True:
            yi,disti = self.__output(u[i,:].reshape(-1,D)) #Salida 
            d_mahal = self.__dmahal(u[i,:].reshape(-1,D)) #Distancia de Mahalanobis 
            self.testDists.append(d_mahal)
            dist_mahal_test.append(disti) ###TEST
            # self.__newEta(yi,err) #Nuevo eta
            err = d[i] - yi # Error
            #Cuantizacion
            min_index = np.argmin(d_mahal)
            
            if d_mahal[min_index] <= self.epsilon:
              self.a_coef[min_index] =(self.a_coef[min_index] + self.eta*err).item()
              self.CB_cov[min_index] = self.__naiveCovQ(min_index,u[i,:])
              self.__n_cov[min_index] = self.__n_cov[min_index] + 1
            else:
              self.CB.append(u[i,:])
              self.a_coef.append((self.eta*err).item())
              self.CB_cov.append(np.eye(D))
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
        # dist = cdist(np.asarray(self.CB), ui)
        dist = self.__dmahal2(ui)
        # print("Dist shape = ", dist.shape)
        K = np.exp(-0.5*(dist**2)) #Quitar sigma  y dist con mhalanobis
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
        dist_m = [distance.mahalanobis(self.__CB_cov_sums[k]/self.__n_cov[k],ui,self.CB_cov[k]/self.__n_cov[k]) for k in range(len(self.CB))]
        dist_m = np.array(dist_m)
        return dist_m

    def __dmahal2(self,ui):
        import numpy as np
        from scipy.spatial import distance
        #List comprehension
        #np.asarray(self.CB)[i]
        dist_m = [distance.mahalanobis(self.CB[k],ui,self.CB_cov[k]/self.__n_cov[k]) for k in range(len(self.CB))]
        dist_m = np.array(dist_m)       
        return dist_m.reshape(-1,1)

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
        return
            
    def predict(self, u):
        import numpy as np
        #Validaciones
        if not self.fitFlag:
            raise ValueError("Fit method must be runed first")
        if u is None:
            raise ValueError("Parameter u is missing")
    
                
        #Tamaños u y d
        N,D = u.shape

        y = np.empty((N,D), dtype=float)
        i = 0
        
        y,d = self.__output(u)
        # while True:           
        #     y[i],disti = self.__output(u[i,:].reshape(-1,D)) #Salida       
            
        #     if(i == N-1):
        #         if self.init_eval:
        #             self.init_eval = False           
        #         return np.array(y)
        #     i+=1
        return y
    
    def score(self, X=None, y=None):
        #Validaciones
        if X is None:
            raise ValueError("Parameter y_true is missing")
        if y is None:
            raise ValueError("Parameter y_pred is missing")
        if len(X) != len(y):
            raise ValueError("All input arguments must be the same lenght, X shape is {0} and y shape is {1}".format(X.shape, y.shape))
                
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        # print("Shapes: \n y:",y.shape," and y_pred:",y_pred.shape)
        return r2_score(y,y_pred)
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"epsilon": self.epsilon, "sigma": self.sigma}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    # def kernelMatrix(self, u):
    #     u=0
        

class QKLMS2:   
    """Filtro QKLMS que aplica distacia de Mahalanobis en la cuantización"""
    def __init__(self, eta=0.9, epsilon=10, sigma=None):
        self.eta = eta #Remplazar por algun criterio
        self.epsilon = epsilon
        self.sigma = sigma
        self.CB = [] #Codebook
        self.a_coef = [] #Coeficientes
        self.CB_cov = [] #Covarianzas
        self.__CB_cov_sums = [] #Sumas acumuladas
        self.__CB_cov_prods = [] #Productos acumulados
        self.n_cov = [] # n iteraciones en covarianza
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
            self.CB_cov.append(np.eye(D)) #Covarianzas
            self.__CB_cov_sums.append(np.zeros((D,)))
            self.__CB_cov_prods.append(np.zeros((D,D)))
            self.n_cov.append(1)
            self.testCB_means.append(np.zeros(2,))#Prueba
            #Salida           
            i = 1
            self.initialize = False
            # err = 0.1
            if u.shape[0] == 1:                
                return
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
              self.CB_cov[min_index] = self.__naiveCovQ(min_index,u[i,:])
              self.n_cov[min_index] = self.n_cov[min_index] + 1
            else:
              self.CB.append(u[i,:])
              self.a_coef.append((self.eta*err).item())
              self.CB_cov.append(np.eye(D))
              self.__CB_cov_sums.append(np.zeros((D,)))
              self.__CB_cov_prods.append(np.zeros((D,D)))
              self.n_cov.append(1)
              
              self.testCB_means.append(np.zeros(2,)) #Prueba
            
            self.CB_growth.append(len(self.CB)) #Crecimiento del diccionario 
            
            if self.init_eval: 
                y[i-1] = yi
            else:
                y[i] = yi

            if(i == N-1):
                # print(len(self.CB_cov))
                # for k in range(len(self.CB_cov)):
                #     print(self.__n_cov[k])
                #     print(self.CB_cov[k]/self.__n_cov[k])
                if self.init_eval:
                    self.init_eval = False           
                return y
            i+=1 

    def __output(self,ui):
        from scipy.spatial.distance import cdist
        import numpy as np
        dist = cdist(np.asarray(self.CB), ui)
        K = np.exp(-0.5*(dist**2)/(self.sigma**2)) #Quitar sigma  y dist con mhalanobis
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
        dist_m = [distance.mahalanobis(self.__CB_cov_sums[k]/self.n_cov[k],ui,self.CB_cov[k]/self.n_cov[k]) for k in range(len(self.CB))]
        dist_m = np.array(dist_m)
        
        return dist_m

    def __naiveCovQ(self,index_cov,ui):
        import numpy as np
        sums = np.asarray(self.__CB_cov_sums[index_cov])
        prods = np.asarray(self.__CB_cov_prods[index_cov])
        n = self.n_cov[index_cov]
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
              self.CB_cov[min_index] = self.__naiveCovQ(min_index,u[i,:])
              self.__n_cov[min_index] = self.__n_cov[min_index] + 1
            else:
              self.CB.append(u[i,:])
              self.a_coef.append((self.eta*err).item())
              self.CB_cov.append(np.eye(D))
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
        return
            
    def predict(self, u):
        import numpy as np
        #Validaciones
        if not self.fitFlag:
            raise ValueError("Fit method must be runed first")
        if u is None:
            raise ValueError("Parameter u is missing")
    
                
        #Tamaños u y d
        N,D = u.shape

        y = np.empty((N,D), dtype=float)
        i = 0
        
        y,d = self.__output(u)
        # while True:           
        #     y[i],disti = self.__output(u[i,:].reshape(-1,D)) #Salida       
            
        #     if(i == N-1):
        #         if self.init_eval:
        #             self.init_eval = False           
        #         return np.array(y)
        #     i+=1
        return y
    
    def score(self, X=None, y=None):
        #Validaciones
        if X is None:
            raise ValueError("Parameter y_true is missing")
        if y is None:
            raise ValueError("Parameter y_pred is missing")
        if len(X) != len(y):
            raise ValueError("All input arguments must be the same lenght, X shape is {0} and y shape is {1}".format(X.shape, y.shape))
                
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        # print("Shapes: \n y:",y.shape," and y_pred:",y_pred.shape)
        return r2_score(y,y_pred)
    
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
                return
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

