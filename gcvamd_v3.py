# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score, f1_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,InputLayer,Conv2DTranspose,UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.initializers import HeNormal, GlorotNormal

tf.__version__ #implemented 2.19.0

import kagglehub

kagglehub.__version__ #0.3.13

import kagglehub

# Download latest version
path = kagglehub.dataset_download("orvile/octdl-optical-coherence-tomography-dataset")

print("Path to dataset files:", path)

amd_dir = '/kaggle/input/octdl-optical-coherence-tomography-dataset/OCTDL/OCTDL/AMD'
nor_dir = '/kaggle/input/octdl-optical-coherence-tomography-dataset/OCTDL/OCTDL/NO'
dt_labels_path= '/kaggle/input/octdl-optical-coherence-tomography-dataset/OCTDL/OCTDL_labels.csv'

#Use this path if above path does not work
amd_dir = '/root/.cache/kagglehub/datasets/orvile/octdl-optical-coherence-tomography-dataset/versions/1/OCTDL/OCTDL/AMD'
nor_dir = '/root/.cache/kagglehub/datasets/orvile/octdl-optical-coherence-tomography-dataset/versions/1/OCTDL/OCTDL/NO'

dt_labels_path = '/root/.cache/kagglehub/datasets/orvile/octdl-optical-coherence-tomography-dataset/versions/1/OCTDL/OCTDL_labels.csv'

dt_labels = pd.read_csv(dt_labels_path)

#checking data
amd_labels = dt_labels.iloc[:1231,:]
nor_labels = dt_labels.iloc[1533:1865,:]
amd_labels.head(5)

np.unique(nor_labels['subcategory'])

map1 = {'emmetropia':0, 'myopia':0,'early':1,'intermediate':2,'late':3}
map2 = {'AMD':1,'NO':0}
amd_labels['status'] = amd_labels['subcategory'].map(map1)
nor_labels['status'] = nor_labels['subcategory'].map(map1)
amd_labels['target'] = amd_labels['disease'].map(map2)
nor_labels['target'] = nor_labels['disease'].map(map2)

amd_labels

amd_labels['MNV'] = amd_labels['condition'].apply(lambda x: 1 if x=='MNV_suspected' else (1 if x=='MNV' else 0))
amd_labels['DN'] = amd_labels['condition'].apply(lambda x: 1 if x=='drusen' else 0)

nor_labels['MNV'] = nor_labels['condition'].apply(lambda x: 1 if x=='MNV_suspected' else (1 if x=='MNV' else 0))
nor_labels['DN'] = nor_labels['condition'].apply(lambda x: 1 if x=='drusen' else 0)

nor_labels

def load_images_and_labels(image, label):
    IMG = []
    LB = []
    ST =[]
    MNV = []
    DN = []
    for index, row in label.iterrows():
        img_path = os.path.join(image, f"{row['file_name']}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224)) # Resize images
            IMG.append(img)
            LB.append(row['target'])
            ST.append(row['status'])
            MNV.append(row['MNV'])
            DN.append(row['DN'])
            if len(LB)%100 == 0: print(index+1)
    return np.array(IMG), np.array(LB), np.array(ST), np.array(MNV), np.array(DN)

# Loading images and labels
X_1, tag1, y_1,nv_1, dn_1 = load_images_and_labels(amd_dir, amd_labels)
X_2, tag2, y_2,nv_2, dn_2 = load_images_and_labels(nor_dir, nor_labels)

tag1

#Example of dataset
plt.imshow(X_1[20,:,:,:])
print(tag1[20],y_1[20],nv_1[20],dn_1[20])

###################

np.random.seed(321) #sampling random numbers for random under sampling
inx = np.random.choice([i for i in range(0,1231,1)],150,replace=False)
inx2 = np.random.choice([i for i in range(0,332,1)],150,replace=False)

#random under sampling
X_ = X_1[inx,:,:,:]

tag_ = tag1[inx]
y_ = y_1[inx]
nv_ = nv_1[inx]
dn_ = dn_1[inx]

X__ = X_2[inx2,:,:,:]

tag__ = tag2[inx2]
y__ = y_2[inx2]
nv__ = nv_2[inx2]
dn__ = dn_2[inx2]

#For downstream task
X_re1 = np.delete(X_1,inx,axis=0)
tag_re1 = np.delete(tag1,inx)
y_re1 = np.delete(y_1,inx)

np.random.seed(789)
inx3 = np.random.choice([i for i in range(0,(1231-150),1)],(332-150),replace=False)
X_re1_ = X_re1[inx3,:,:,:]
tag_re1_ = tag_re1[inx3]
y_re1_ = y_re1[inx3]

X_re2 = np.delete(X_2,inx2,axis=0)
tag_re2 = np.delete(tag2,inx2)
y_re2 = np.delete(y_2,inx2)

#For downstream task

X_te1 = np.concatenate([X_re1_,X_re2],axis=0)
tag_te1 = np.concatenate([tag_re1_,tag_re2],axis=0)
y_te1 = np.concatenate([y_re1_,y_re2],axis=0)

X_te1 = X_te1/255.0

(X_te2,X_te3,tag_te2,tag_te3,y_te2,y_te3) = train_test_split(X_te1,tag_te1,y_te1,test_size=0.3,shuffle=True,random_state=321)
X_te = np.concatenate([X_te2,X_te3],axis=0)
tag_te = np.concatenate([tag_te2,tag_te3])
y_te = np.concatenate([y_te2,y_te3])

#For causal learning
X_tr1 = np.concatenate([X_,X__],axis=0) #concatenating with minority class(Normal) data
tag_tr1 = np.concatenate([tag_,tag__],axis=0)
y_tr1 = np.concatenate([y_,y__],axis=0)
nv_tr1 = np.concatenate([nv_,nv__],axis=0)
dn_tr1 = np.concatenate([dn_,dn__],axis=0)

###Train data for CausalVAE

X_tr1 = X_tr1/255.0 #rescaling

#reshuffling data
(X_tr2,X_tr3,tag_tr2,tag_tr3,y_tr2,y_tr3,nv_tr2,nv_tr3,dn_tr2,dn_tr3) = train_test_split(X_tr1,tag_tr1,y_tr1,nv_tr1,dn_tr1,test_size=0.2,shuffle=True,random_state=321)
X_tr = np.concatenate([X_tr2,X_tr3],axis=0)
tag_tr = np.concatenate([tag_tr2,tag_tr3],axis=0)
y_tr = np.concatenate([y_tr2,y_tr3],axis=0)
nv_tr = np.concatenate([nv_tr2,nv_tr3],axis=0)
dn_tr = np.concatenate([dn_tr2,dn_tr3],axis=0)
print(np.bincount(y_tr))
print(np.bincount(tag_tr))

plt.imshow(X_tr[6,:,:,:])

################Defining GCVAMD

#CausalVAE+GAE Structures

tf.keras.utils.set_random_seed(321)

class AMD_VAE(tf.keras.Model):
  def __init__(self,input_dim, latent_dim,h1,h2,h3,d1,d1_2,d2):
    super(AMD_VAE, self).__init__()
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.h1 = h1
    self.h2 = h2
    self.h3 = h3
    self.d1 = d1
    self.d1_2 = d1_2
    self.d2 = d2


    self.ADJ = self.add_weight(shape=(self.latent_dim,self.latent_dim),initializer="zeros",trainable=True,name="adj") #"random_normal"

    k = tf.keras.initializers.HeNormal(123)
    self.Enc = Sequential([
        InputLayer(shape=(input_dim,input_dim,3)),
        Conv2D(filters=h1,kernel_size=5,strides=(3,3),activation="silu"),
        Conv2D(filters=h2,kernel_size=4,strides=(2,2),activation="silu"),
        Conv2D(filters=h3,kernel_size=4,strides=(2,2),activation="silu"),
        Flatten(),
        Dense(units = d1,activation="elu",kernel_initializer=k),
        Dense(units = d1_2,activation="elu",kernel_initializer=k),
        Dense(units=latent_dim+latent_dim,kernel_initializer=k,activation="linear")
        ])

    self.Dec = Sequential([
        InputLayer(shape=(latent_dim,)),
        Dense(units=d1_2, activation="elu",kernel_initializer=k),
        Dense(units = d1,activation="elu",kernel_initializer=k),
        Dense(units=17*17*h3, activation="elu",kernel_initializer=k),
        tf.keras.layers.Reshape(target_shape=(17,17,h3)),
        Conv2DTranspose(filters=h2, kernel_size=4, strides=(2,2), activation='silu',kernel_initializer=k),
        Conv2DTranspose(filters=h1, kernel_size=4, strides=(2,2),activation='silu',kernel_initializer=k),
        Conv2DTranspose(filters=3, kernel_size=5, strides=(3,3), kernel_initializer=k,activation='sigmoid'),
        ])



    mat1 = np.zeros(latent_dim*latent_dim*d2).reshape(latent_dim,latent_dim*d2)
    mat2 = np.zeros(latent_dim*d2*latent_dim*d2).reshape(latent_dim*d2,latent_dim*d2)
    mat3 = np.zeros(latent_dim*d2*latent_dim).reshape(latent_dim*d2,latent_dim)

    for i in range(latent_dim):
      mat1[i,(d2*i):(d2*(i+1))] = 1
      mat2[(i*d2):((i+1)*d2),(d2*i):(d2*(i+1))] = 1
      mat3[(i*d2):((i+1)*d2),i] = 1

    mask1 = np.array(mat1).reshape(latent_dim,latent_dim*d2)
    class mask_1(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask1)*w


    mask2 = np.array(mat2).reshape(latent_dim*d2,latent_dim*d2)
    class mask_2(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask2)*w

    mask3 = np.array(mat3).reshape(latent_dim*d2,latent_dim)
    class mask_3(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask3)*w


    k2 = tf.keras.initializers.HeNormal(123)
    self.G_ENC = Sequential([
        InputLayer(shape=(latent_dim,)),
        Dense(units = latent_dim*d2, use_bias=False, kernel_initializer=k2,activation="elu",kernel_constraint=mask_1()),
        Dense(units = latent_dim*d2, use_bias=False,kernel_initializer=k2,activation="elu",kernel_constraint=mask_2()),
        Dense(units=latent_dim,use_bias=False,kernel_initializer=k2,activation="linear",kernel_constraint=mask_3())
        ])

    self.G_DEC = Sequential([
        InputLayer(shape=(latent_dim,)),
        Dense(units = latent_dim*d2, use_bias=False,kernel_initializer=k2,activation="elu",kernel_constraint=mask_1()),
        Dense(units = latent_dim*d2, use_bias=False,kernel_initializer=k2,activation="elu",kernel_constraint=mask_2()),
        Dense(units=latent_dim,use_bias=False,kernel_initializer=k2,activation="linear",kernel_constraint=mask_3())
        ])

    k3 = tf.keras.initializers.HeNormal(123)
    self.eps_ENC = Sequential([
        InputLayer(shape=(latent_dim,)),
        Dense(units = latent_dim*d2, use_bias=False, kernel_initializer=k3,activation="elu",kernel_constraint=mask_1()),
        Dense(units = latent_dim*d2, use_bias=False,kernel_initializer=k3,activation="elu",kernel_constraint=mask_2()),
        Dense(units=latent_dim,use_bias=False,kernel_initializer=k3,activation="linear",kernel_constraint=mask_3())
        ])

    self.eps_DEC = Sequential([
        InputLayer(shape=(latent_dim,)),
        Dense(units = latent_dim*d2, use_bias=False,kernel_initializer=k3,activation="elu",kernel_constraint=mask_1()),
        Dense(units = latent_dim*d2, use_bias=False,kernel_initializer=k3,activation="elu",kernel_constraint=mask_2()),
        Dense(units=latent_dim,use_bias=False,kernel_initializer=k3,activation="linear",kernel_constraint=mask_3())
        ])




  #@tf.function

  def enc(self, x): #returns encoded noise eps
    ec = self.Enc(x,training=True)
    mean, lv = tf.split(ec,num_or_size_splits=2,axis=1)
    return mean, lv #lv: log-variance


  def reparam(self, mean, lv):
    eps = tf.random.normal(shape=mean.shape) #seed not fixed for randomness
    return eps*tf.math.exp(lv*0.5) + mean #lv: log variance

  def dec(self,z, sigmoid=False):
    result = self.Dec(z,training=True)
    if sigmoid == True:
        result = tf.math.sigmoid(result)
        return result
    return result


  def eps_to_z(self,eps):
    eps1 = self.eps_ENC(eps,training=True)
    eps1_T = tf.transpose(eps1)
    I = tf.eye(self.latent_dim, dtype=self.ADJ.dtype)
    M = I-self.ADJ
    eps2_T = tf.linalg.solve(M, eps1_T)
    eps2 = tf.transpose(eps2_T)
    z = self.eps_DEC(eps2,training=True)
    return z

  mse_loss = tf.keras.losses.MeanSquaredError()
  def gae_loss(self,x):
    mean, lv = self.enc(x)
    eps = self.reparam(mean, lv)
    z = self.eps_to_z(eps)
    z1 = self.G_ENC(z,training=True)
    z2 = tf.linalg.matmul(z1, self.ADJ)
    z_hat = self.G_DEC(z2,training=True)
    return mse_loss(z,z_hat)

  def to_z_hat(self,z):
    z1 = self.G_ENC(z,training=True)
    z2 = tf.linalg.matmul(z1, self.ADJ)
    z_hat = self.G_DEC(z2,training=True)
    return z_hat

  mse_loss2 = tf.keras.losses.MeanSquaredError()
  b_loss2 = tf.keras.losses.BinaryCrossentropy()
  def C_ELBO_loss(model, x): #ELBO loss
    mean, lv = model.enc(x)
    eps = model.reparam(mean, lv)
    z = model.eps_to_z(eps)
    z_hat = model.to_z_hat(z)
    x_hat = model.dec(z_hat, sigmoid=False)
    flat = tf.keras.layers.Flatten()
    f_x_hat = flat(x_hat)
    f_x = flat(x)
    loss_2 = tf.norm(eps, ord=2)
    return b_loss2(f_x,f_x_hat)+loss_2/x.shape[0]

  mse_loss3 = tf.keras.losses.MeanSquaredError()
  def u_loss(self, u): #U variable loss
    u_hat = tf.linalg.matmul(u,self.ADJ)
    return mse_loss3(u, u_hat)

  mse_loss4 = tf.keras.losses.MeanSquaredError()
  def zu_loss(self, x,u):
    mean, lv = self.enc(x)
    eps = self.reparam(mean, lv)
    z = self.eps_to_z(eps)
    return mse_loss4(u,z)

#Label Data
U_data = pd.DataFrame({'u1':nv_tr,'u2':dn_tr,'u3':y_tr})
U_data.shape

#Updating GCVAMD model

tf.keras.utils.set_random_seed(987)
os.environ['TF_DETERMINISTIC_OPS']='1'

tf.executing_eagerly()
import tensorflow.keras.backend as K

Epochs = 250 #250
lat_dim_ = 3
GCVAMD = AMD_VAE(224,3,16,16,32,256,64,4)
mse_loss = tf.keras.losses.MeanSquaredError()
mse_loss2 = tf.keras.losses.MeanSquaredError()
mse_loss3 = tf.keras.losses.MeanSquaredError()
mse_loss4 = tf.keras.losses.MeanSquaredError()
b_loss2 = tf.keras.losses.BinaryCrossentropy()


alpha=0.6 #
i = 0
rho = 0.1 #
gamma=0.9
beta = 1.01
lamb = 1.0 #L1-regularization


loss_of_cs = []
loss_of_cv = []
basic_opt = tf.keras.optimizers.Adam(learning_rate=0.02)
basic_opt2 = tf.keras.optimizers.Adam(learning_rate=0.002)
basic_opt3 = tf.keras.optimizers.Adam(learning_rate=0.003)
basic_opt4 = tf.keras.optimizers.Adam(learning_rate=0.002)
U_data = tf.reshape(tf.convert_to_tensor(U_data,dtype=tf.float32),[300,3])

while i < Epochs:
    with tf.GradientTape() as dv_t, tf.GradientTape() as dv_t2, tf.GradientTape() as dv_t3, tf.GradientTape() as dg_t:
      loss_ = GCVAMD.C_ELBO_loss(X_tr)
      loss_2 = GCVAMD.gae_loss(X_tr)
      loss_3 = GCVAMD.u_loss(U_data)
      loss_4 = GCVAMD.zu_loss(X_tr,U_data)
      h_a = tf.linalg.trace(tf.linalg.expm(tf.math.multiply(GCVAMD.weights[0], GCVAMD.weights[0])))-lat_dim_ #
      if i < 100:
        cs_l = loss_ + 0.3*loss_2 + 0.3*loss_3+ alpha*h_a+rho*0.5*tf.math.abs(h_a)**2 +0.1*loss_4
        d_l = 0.3*loss_+ 2*loss_2 + 0.5*loss_3 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2 +0.1*loss_4
      else:
        cs_l = loss_ + 0.3*loss_2 + 0.3*loss_3+ alpha*h_a+rho*0.5*tf.math.abs(h_a)**2 +0.1*loss_4
        d_l = 0.3*loss_+ 2*loss_2 + 0.5*loss_3 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2 +0.3*loss_4
    loss_of_cs.append(cs_l)
    loss_of_cv.append(d_l)
    grad_g_a = dv_t.gradient(d_l, [GCVAMD.trainable_variables[0]])
    grad_g_b = dv_t2.gradient(cs_l, GCVAMD.trainable_variables[1:21])
    grad_g_c = dv_t3.gradient(cs_l, GCVAMD.trainable_variables[27:])
    grad_g = dg_t.gradient(d_l, GCVAMD.trainable_variables[21:27])
    basic_opt.apply_gradients(zip(grad_g_a, [GCVAMD.trainable_variables[0]]))
    basic_opt2.apply_gradients(zip(grad_g_b, GCVAMD.trainable_variables[1:21]))
    basic_opt4.apply_gradients(zip(grad_g_c,GCVAMD.trainable_variables[27:]))
    basic_opt3.apply_gradients(zip(grad_g, GCVAMD.trainable_variables[21:27]))
    h_a_new = tf.linalg.trace(tf.linalg.expm(tf.math.multiply(GCVAMD.weights[0], GCVAMD.weights[0])))-lat_dim_
    alpha =  alpha + rho * h_a_new
    if (tf.math.abs(h_a_new) >= gamma*tf.math.abs(h_a)):
        rho = beta*rho
    else:
        rho = rho
    if (i+1) %10 == 0: print(i+1, cs_l,d_l,loss_)
    if (i+1) == 100: basic_opt = tf.keras.optimizers.Adam(learning_rate=0.04)
    i = i+1

###########################

#causal weighted adjacency matrix(fitted)
import seaborn as sns
sns.set_style("darkgrid")
sns.heatmap(np.array(GCVAMD.weights[0]), cmap="vlag",center=0)
plt.show()

#binarized adjacency matrix
import seaborn as sns
sns.set_style("darkgrid")
sns.heatmap(np.where(np.abs(np.array(GCVAMD.weights[0]))> np.quantile(np.abs(np.array(GCVAMD.weights[0])),0.8),1,0), cmap="gray_r",linewidths=1,linecolor="black")
plt.show()

mean, lv = GCVAMD.enc(X_tr)
eps = GCVAMD.reparam(mean, lv)
z = GCVAMD.eps_to_z(eps)
z_hat = GCVAMD.to_z_hat(z)
x_hat = GCVAMD.dec(z_hat, sigmoid=False)

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_cv, color="black",label="Loss_2")
#plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_cs, color="red",label="Loss_1")
#plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

d_l #0.5813837051391602

loss_of_cs[-1] #0.6881941556930542

print(loss_,loss_2,loss_3)

#######################Quantitative Check for disentanglement

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler



tf.keras.utils.set_random_seed(456)
mean, lv = GCVAMD.enc(X_tr)
eps = GCVAMD.reparam(mean, lv)
z = GCVAMD.eps_to_z(eps)

U_dat = np.array(U_data)

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

#MI checking
mutual_info_regression(np.array(z[:,0]).reshape(-1,1),U_dat[:,0],n_neighbors=5, random_state=3)



#Standardizing to have 0 mean and unit variance
st = StandardScaler()
z_st = st.fit_transform(np.array(z))
np.var(z_st)
st2 = StandardScaler()
U_dat_st = st2.fit_transform(np.array(U_dat))
np.var(U_dat_st)
u0 = U_dat_st[:,0]
u1 = U_dat_st[:,1]
u2 = U_dat_st[:,2]

ls_1 = Lasso(alpha=0.1,random_state=123)
ls_1.fit(z_st,u0)

ls_2 = Lasso(alpha=0.1,random_state=123)
ls_2.fit(z_st,u1)

ls_3 = Lasso(alpha=0.01,random_state=123)
ls_3.fit(z_st,u2)

Prob_mat = np.zeros(3*3).reshape(3,3)
Prob_mat[0,:] = np.abs(ls_1.coef_)
Prob_mat[1,:] = np.abs(ls_2.coef_)
Prob_mat[2,:] = np.abs(ls_3.coef_)
Prob_mat_fac = Prob_mat*1
for q1 in range(3):
  for q2 in range(3):
    if Prob_mat_fac[q1,q2] == 0:
      Prob_mat_fac[q1,q2] += 1e-4
Prob_mat_fac[0,:] = Prob_mat_fac[0,:]/sum(Prob_mat_fac[0,:])
Prob_mat_fac[1,:] = Prob_mat_fac[1,:]/sum(Prob_mat_fac[1,:])
Prob_mat_fac[2,:] = Prob_mat_fac[2,:]/sum(Prob_mat_fac[2,:])
Prob_mat_fac

#Compactness metric
c0 = Prob_mat_fac[0,:]
C0 = 1+np.sum(c0*np.emath.logn(3,c0));C0

c1 = Prob_mat_fac[1,:]
C1 = 1+np.sum(c1*np.emath.logn(3,c1));C1

c2 = Prob_mat_fac[2,:]
C2 = 1+np.sum(c2*np.emath.logn(3,c2));C2

print(C0,C1,C2)

np.mean([C0,C1,C2])

print(ls_1.coef_,ls_2.coef_,ls_3.coef_)

#Disentanglement Metric
Prob_mat = np.zeros(3*3).reshape(3,3)
Prob_mat[0,:] = np.abs(ls_1.coef_)
Prob_mat[1,:] = np.abs(ls_2.coef_)
Prob_mat[2,:] = np.abs(ls_3.coef_)
Prob_mat_code = Prob_mat*1
for q1 in range(3):
  for q2 in range(3):
    if Prob_mat_code[q1,q2] == 0:
      Prob_mat_code[q1,q2] += 1e-5
for i in range(3):
  Prob_mat_code[:,i] = Prob_mat_code[:,i]/sum(Prob_mat_code[:,i])
Prob_mat_code #

p0 = Prob_mat_code[:,0]
D0 = 1+np.sum(p0*np.emath.logn(3,p0));D0

p1 = Prob_mat_code[:,1]
D1 = 1+np.sum(p1*np.emath.logn(3,p1));D1

p2 = Prob_mat_code[:,2]
D2 = 1+np.sum(p2*np.emath.logn(3,p2));D2

print(D0,D1,D2) #disentanglement score

np.mean([D0,D1,D2])



####################Qualitative Checks for disentanglement

z_hat = GCVAMD.to_z_hat(z)
x_hat = GCVAMD.dec(z_hat, sigmoid=False)

#Disentanglement simulation (Qualitative)
set1 = [-0.2,-0.1,-0.05,0,0.05,0.1,0.2] #

fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=7
ax3 = []

#Modifying only individual Z with other latent variables fixed.
for i in range(7):
  distangle = np.array(z[53,:])
  distangle[0] = set1[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,3])
  distg = GCVAMD.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z0:"+str(set1[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,2])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#Disentanglement simulation (Qualitative)
set1 = [-0.2,-0.1,-0.05,0,0.05,0.1,0.2] #

fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=7
ax3 = []

#Modifying only individual Z with other latent variables fixed.
for i in range(7):
  distangle = np.array(z[205,:])
  distangle[0] = set1[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,3])
  distg = GCVAMD.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z0:"+str(set1[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,2])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#Disentanglement simulation (Qualitative)
set1 = [-0.2,-0.1,-0.05,0,0.05,0.1,0.2] #

fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=7
ax3 = []

#Modifying only individual Z with other latent variables fixed.
for i in range(7):
  distangle = np.array(z[123,:])
  distangle[0] = set1[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,3])
  distg = GCVAMD.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z0:"+str(set1[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,2])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

###################

#Disentanglement simulation (Qualitative)
set2 = [-0.1,-0.05,-0.025,0,0.025,0.05,0.15] #

fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=7
ax3 = []

#Modifying only individual Z with other latent variables fixed.
for i in range(7):
  distangle = np.array(z[246,:])
  distangle[1] = set2[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,3])
  distg = GCVAMD.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z1:"+str(set2[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,2])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#Disentanglement simulation (Qualitative)
set2 = [-0.1,-0.05,-0.025,0,0.025,0.05,0.15] #

fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=7
ax3 = []

#Modifying only individual Z with other latent variables fixed.
for i in range(7):
  distangle = np.array(z[139,:])
  distangle[1] = set2[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,3])
  distg = GCVAMD.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z1:"+str(set2[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,2])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#Disentanglement simulation (Qualitative)
set2 = [-0.1,-0.05,-0.025,0,0.025,0.05,0.15] #

fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=7
ax3 = []

#Modifying only individual Z with other latent variables fixed.
for i in range(7):
  distangle = np.array(z[72,:])
  distangle[1] = set2[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,3])
  distg = GCVAMD.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z1:"+str(set2[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,2])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#Disentanglement simulation (Qualitative)
set3 = [-0.1,-0.05,0,0.05,0.1,0.2,0.3]

fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=7
ax3 = []

#Modifying only individual Z with other latent variables fixed.
for i in range(7):
  distangle = np.array(z_hat[128,:])
  distangle[2] = set3[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,3])
  distg = GCVAMD.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z2:"+str(set3[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,2])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

plt.imshow(X_tr[53,:,:,:])#Original image.
print(tag_tr[53],dn_tr[53],nv_tr[53])
plt.grid()
plt.show()

plt.imshow(X_tr[246,:,:,:])#Original image.
print(tag_tr[246],dn_tr[246],nv_tr[246])
plt.grid()
plt.show()

plt.imshow(X_tr[72,:,:,:])#Original image. 139,72
print(tag_tr[72],dn_tr[72],nv_tr[72])
plt.grid()
plt.show()

plt.imshow(X_tr[123,:,:,:])#Original image. 205, 123
print(tag_tr[123],dn_tr[123],nv_tr[123])
plt.grid()
plt.show()

np.array([i for i in range(300) if dn_tr[i]==1]).reshape(-1)

np.array([i for i in range(300) if nv_tr[i]==1]).reshape(-1)

np.bincount(nv_tr)

plt.imshow(X_tr[128,:,:,:])#Original image.
print(tag_tr[128],dn_tr[128],nv_tr[128])
plt.grid()
plt.show()

#Downstream task experiments: encoding train, test data each for analysis based on CVAE.
tf.keras.utils.set_random_seed(789)

m_1, l_1 = GCVAMD.enc(X_tr)
eps1 = GCVAMD.reparam(m_1, l_1)
z_tr1 = GCVAMD.eps_to_z(eps1)
z_tr2 = GCVAMD.to_z_hat(z_tr1) #

tf.keras.utils.set_random_seed(789)

m_te1, l_te1 = GCVAMD.enc(X_te)
eps2 = GCVAMD.reparam(m_te1, l_te1)
z_te1 = GCVAMD.eps_to_z(eps2)
z_te2 = GCVAMD.to_z_hat(z_te1) #



####Convolutional Vanilla AE

tf.keras.utils.set_random_seed(987)

input1 = tf.keras.layers.Input(shape=(224,224,3))
x = Conv2D(filters=16,kernel_size=5,strides=(3,3),activation="silu")(input1)
x = Conv2D(filters=16,kernel_size=4,strides=(2,2),activation="silu")(x)
x = Conv2D(filters=32,kernel_size=4,strides=(2,2),activation="silu")(x)
x = tf.keras.layers.Flatten()(x)
x = Dense(units = 256,activation="elu")(x) #
x = Dense(units = 64,activation="elu")(x) #
y_ = Dense(units=10,activation="linear")(x) #
Encoder = tf.keras.models.Model(inputs=input1, outputs=y_)


input2 = tf.keras.layers.Input(shape=(10,))
x2 = Dense(units=10,activation="linear")(input2)
x2 = Dense(units = 64,activation="elu")(x2)
x2 = Dense(units = 256,activation="elu")(x2)
x2 = Dense(units=17*17*32, activation="elu")(x2)
x2 = tf.keras.layers.Reshape(target_shape=(17,17,32))(x2)
x2 = Conv2DTranspose(filters=16, kernel_size=4, strides=(2,2), activation='silu')(x2)
x2 = Conv2DTranspose(filters=16, kernel_size=4, strides=(2,2),activation='silu')(x2)
y_2 = Conv2DTranspose(filters=3, kernel_size=5, strides=(3,3), activation='sigmoid')(x2)
Decoder = tf.keras.models.Model(inputs=input2, outputs=y_2)

tf.keras.utils.set_random_seed(987)
class AE(tf.keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AE()

autoencoder.encoder.summary()

tf.keras.utils.set_random_seed(987)
from tensorflow.keras.optimizers import RMSprop, Adam
autoencoder.compile(optimizer=Adam(learning_rate=5e-3), loss=tf.keras.losses.BinaryCrossentropy()) #0.002

tf.keras.utils.set_random_seed(987)
hist_ = autoencoder.fit(X_tr, X_tr,epochs=300, batch_size=100, shuffle=True)

rec = autoencoder.decoder(autoencoder.encoder(X_tr))
plt.imshow(rec[10,:,:,:])
plt.grid()

autoencoder.evaluate(X_tr,X_tr)

#non-causal data from AE
tf.keras.utils.set_random_seed(987)
non_c_tr = np.array(autoencoder.encoder(X_tr))
non_c_te = np.array(autoencoder.encoder(X_te))

non_c_te.shape

#Concatenating causal data to non-causal data
z_tr = np.concatenate([z_tr1[:,0:2],np.array(z_tr2[:,2]).reshape(-1,1),non_c_tr],axis=1)
z_te = np.concatenate([z_te1[:,0:2],np.array(z_te2[:,2]).reshape(-1,1),non_c_te],axis=1)

#DNN for predicting AMD
tf.keras.utils.set_random_seed(321321)
os.environ['TF_DETERMINISTIC_OPS']='1'
k3 = tf.keras.initializers.HeNormal(123)

md_c = Sequential([
    InputLayer(shape=(13,)),
    BatchNormalization(),
    Dense(units = 32, kernel_initializer=k3, activation="elu"),
    BatchNormalization(),
    Dense(units = 16, kernel_initializer=k3, activation="elu"),#
    BatchNormalization(),#
    Dense(units = 4, kernel_initializer=k3, activation="elu"),
    BatchNormalization(),
    Dense(units=1,kernel_initializer=k3, activation="sigmoid")
    ])

np.bincount(tag_tr)

from tensorflow.keras.optimizers import RMSprop, Adam
md_c.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"]) #1e-4

hist2 = md_c.fit(x=z_tr, y=tag_tr,epochs=400,batch_size=50)

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(hist2.history['loss'], color="red",label="loss")

f1_score(tag_tr,np.round(md_c.predict(z_tr)))

accuracy_score(tag_tr, np.round(md_c.predict(z_tr)))

#Prediction results
prd = md_c.predict(z_te)
prd1 = []
for i in range(len(tag_te)):
  if prd[i] < 0.5:
    prd1.append(0)
  else:
    prd1.append(1)

from sklearn.metrics import ConfusionMatrixDisplay
dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(tag_te,prd1), display_labels=[0,1])
dis.plot()
plt.grid()
plt.show()

from sklearn.metrics import classification_report

print(classification_report(tag_te,prd1,digits=4))

from sklearn.metrics import precision_score, recall_score,f1_score
print(precision_score(tag_te,prd1))
print(recall_score(tag_te,prd1))
print(f1_score(tag_te,prd1,average="macro"))

from sklearn.metrics import roc_auc_score

roc_auc_score(tag_te,md_c.predict(z_te))

#DNN for predicting AMD
tf.keras.utils.set_random_seed(321321)
os.environ['TF_DETERMINISTIC_OPS']='1'
k3 = tf.keras.initializers.HeNormal(123)

md_d = Sequential([
    InputLayer(shape=(10,)),
    BatchNormalization(),
    Dense(units = 32, kernel_initializer=k3, activation="elu"),
    BatchNormalization(),
    Dense(units = 16, kernel_initializer=k3, activation="elu"),#
    BatchNormalization(),#
    Dense(units = 4, kernel_initializer=k3, activation="elu"),
    BatchNormalization(),
    Dense(units=1,kernel_initializer=k3, activation="sigmoid")
    ])

from tensorflow.keras.optimizers import RMSprop, Adam
md_d.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

hist3 = md_d.fit(x=non_c_tr, y=tag_tr,epochs=400,batch_size=50)

#Prediction results
prd2_ = md_d.predict(non_c_te)
prd2 = []
for i in range(len(tag_te)):
  if prd2_[i] < 0.5:
    prd2.append(0)
  else:
    prd2.append(1)

from sklearn.metrics import ConfusionMatrixDisplay
dis2 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(tag_te,prd2), display_labels=[0,1])
dis2.plot()
plt.grid()
plt.show()

print(classification_report(tag_te,prd2,digits=4))

from sklearn.metrics import precision_score, recall_score,f1_score
print(precision_score(tag_te,prd2))
print(recall_score(tag_te,prd2))
print(f1_score(tag_te,prd2,average="macro"))

#ROC AUC results
from sklearn.metrics import roc_auc_score, roc_curve

fpr1, tpr1, _1 = roc_curve(tag_te, md_c.predict(z_te))
fpr2, tpr2, _2 = roc_curve(tag_te,md_d.predict(non_c_te))
plt.plot(fpr1,tpr1, label="Test_Causal+Non-Causal"+" area:"+str(np.round(roc_auc_score(tag_te,md_c.predict(z_te)),4)),color="red")
plt.plot(fpr2,tpr2, label="Test_Non-Causal"+" area:"+str(np.round(roc_auc_score(tag_te,md_d.predict(non_c_te)),4)),color="blue",linestyle="--")
plt.plot(np.linspace(0,1,1000),np.linspace(0,1,1000),color="black",linestyle="--")
plt.legend()
#plt.grid()
plt.show()

roc_auc_score(tag_te,md_d.predict(non_c_te))

