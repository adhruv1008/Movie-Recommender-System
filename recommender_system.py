#importing libraries
import pandas as pd
import numpy as np
import torch
from time import time

#importing data
movies = pd.read_csv("dataset/movies.dat", sep = "::", header = None, engine = "python", encoding = "latin-1")
training_set = pd.read_csv("dataset/ratings.dat", sep = "::", header = None, engine = "python", encoding = "latin-1")
training_set = np.array(training_set)

# Getting the number of users and movies
nb_users = int(max(training_set[:,0]))
nb_movies = int(max(training_set[:,1]))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)

#converting the data into torch tensors
training_set = torch.FloatTensor(training_set)

#converting data into binary format : 0(dislike) , 1(like)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

#Creating the architecture of RBM
class RBM():
    def __init__(self, nv, nh):
        #initializing weights
        self.W = torch.randn(nh, nv)
        #initializing bias for getting probability of hidden nodes given visible nodes
        self.a = torch.randn(1, nh)
        #initializing bias for getting probability of visible nodes given hidden nodes
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.a += torch.sum((ph0 - phk), 0)
        self.b += torch.sum((v0 - vk), 0)

nv = len(training_set[0])

#choosing the number of hidden nodes
nh = 100

#choosing the batch_size = number of training examples after which weights should get updated
batch_size = 64

#choosing the number of steps for contrastive divergence
gibbs_sampling = 20

rbm = RBM(nv, nh)

#Training the network
print("training the network .... \n")
t0 = time()

nb_epochs = 10
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    counter = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user: id_user + batch_size]
        v0 = training_set[id_user: id_user + batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(gibbs_sampling):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        counter += 1.
    print("epoch: " + str(epoch) + " ", train_loss/counter)

print("training took ", round((time()-t0)*1.0/3600, 3), "hrs\n")
        
#Loading the test data
test_set = pd.read_csv("dataset/test.dat", delimiter = '\t')
test_set = np.array(test_set, dtype = 'int64')

test_users = int(max(test_set[:,0]))

test_set = convert(test_set)

test_set = torch.FloatTensor(test_set)

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#Testing the model on Test data
print("Testing the data on 80000 ratings provided by 943 users\n")
test_loss = 0
counter = 0.
for id_user in range(test_users):
    v = training_set[id_user: id_user + 1]
    vt = test_set[id_user: id_user + 1]
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        counter += 1.

print("test loss: " + str(test_loss/counter))
        
        
        
        
        