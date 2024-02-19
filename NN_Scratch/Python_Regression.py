import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

# %%
df = pd.read_csv("NN_Scratch/heart.csv")
df
#%%

X = np.array(df.loc[:,df.columns != 'output'])
Y = np.array(df['output'])

print (f"{X.shape}, {Y.shape}")

# %%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.2,random_state = 32 )

#%%
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# %%
class NeuralNetworkFromScratch:
    def __init__(self,LR,X_train,Y_train,X_test,Y_test):
        self.w = np.random.randn(X_train_scale.shape[1])
        self.b = np.random.randn()
        self.LR = LR
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.L_train = []
        self.L_test = []

    def activation(self,x):
        # sigmoid functino
        return 1/(1+np.exp(-x))

    def deriv_activation(self,x):
        #derivative of sigmoid
        return self.activation(x)*(1-self.activation(x))

    def forward(self,x):
        hidden_1 = np.dot(x,self.w) + self.b
        activate_1 = self.activation(hidden_1)

        return activate_1

    def backward(self,X,y_true):
        # calc gradients
        hidden_1 = np.dot(X,self.w) + self.b
        y_pred = self.forward(X)

        dL_dpred = 2 * (y_pred-y_true)
        dpred_dhidden_1 = self.deriv_activation(hidden_1)
        dhidden_1_db = 1
        dhidden_1_dw = X

        dL_db = dL_dpred * dpred_dhidden_1 * dhidden_1_db
        dL_dw = dL_dpred * dpred_dhidden_1 * dhidden_1_dw

        return dL_db,dL_dw

    def optimzier(self,dL_dB,dL_dW):
        self.b = self.b - dL_dB*self.LR
        self.w = self.w - dL_dW*self.LR

    def train(self,iterations ):
        for i in range(iterations):
            # random position
            random_pos = np.random.randint(len(self.X_train))

            y_train_true = self.Y_train[random_pos]
            y_train_pred = self.forward(self.X_train[random_pos])

            # loss
            L = np.square(y_train_pred-y_train_true)
            self.L_train.append(L)

            #calculate gradients
            dL_db,dL_dw = self.backward(self.X_train[random_pos],self.Y_train[random_pos])

            #update weights
            self.optimzier(dL_db,dL_dw)

            # calc error for test data
            L_sum = 0
            for j in range(len(self.X_test)):
                y_true = self.Y_test[j]
                y_pred = self.forward(self.X_test[j])
                L_sum += np.square(y_pred - y_true)

            self.L_test.append(L_sum)

        return "training successful"
# %% Hyper params
LR =  0.1
iterations = 1000

# %%
nn  = NeuralNetworkFromScratch(LR=LR,X_train=X_train_scale,
                               Y_train=Y_train,
                               X_test= X_test_scale,
                               Y_test=Y_test)
nn.train(iterations)

# %%
%matplotlib inline
sns.lineplot(x = list(range(len(nn.L_test))),y=nn.L_test)