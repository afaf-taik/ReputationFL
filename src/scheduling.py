import numpy as np 
from gain import channel_gain 
import math 
from options import args_parser 
import utils
from options import args_parser 

# #Size of the model
S = 160000
# #Total Bandwidth = 1Mhz
B = 1000000
# #inverse of N0 
N10 = 0.25*10**15
T = 300
P = 0.2
args = args_parser()
utils.exp_details(args)


def train_time(args, user_groups,t_batch) :
	n_users = int(args.num_users/2)
	Ttrain = np.zeros(args.num_users)
	t = t_batch*args.local_ep/(args.local_bs)
	for i in range(args.num_users):
		
		Ttrain[i] = t*len(user_groups[i])
	return Ttrain

# def tup(alpha,args,L):
# 	tup=[]
# 	n_users = int(args.num_users/2)
# 	for i in L:
# 		tup.append(S/(alpha*B*(math.log2(1+P*gain[0,i,0]*N10/(alpha[i]*B)))))
# 	return tup	

def datarate(alpha,gain):
	return alpha*B*(math.log2(1+P*gain*N10/(alpha*B)))

# Greedy
def required_bandwidth(T, Ttrain, args, xmax = 500, ymax = 500):
	nbr_fractions = np.ones(args.num_users)
	xyUEs = np.vstack((np.random.uniform(low=0.0, high=xmax, size=args.num_users),np.random.uniform(low=0.0, high=ymax, size=args.num_users))) 
	xyBSs = np.vstack((250,250))
	#CHANNEL GAIN
	Gains = channel_gain(args.num_users,1,1, xyUEs, xyBSs,0)
	gains = []
	for i in range(args.num_users):
		gains.append(Gains[0,i,0])

	cost = np.zeros(args.num_users)
	for k in range(args.num_users):
		rmin = S/(T-Ttrain[k])
		c = 1
		r = 0
		print('required_bandwidth for', k)
		while(r<rmin and c<args.num_users):
			r = datarate(c/args.num_users, gains[i])
			c = c+1
		cost[k] = c
		print(cost[k])			
	return cost	
def allocate(rep,cost):
	priority = [rep[i]/cost[i] for i in range(len(rep))]
	ordered = utils.get_order(priority)
	b = args.num_users 
	k = 0
	while(b>0):
		b = b-cost[k]
		k+=1
	return ordered[:k]		

# def required_bandwidth(T, Ttrain, args, L, xmax = 500, ymax = 500):
# 	n_users = int(args.num_users/2)
# 	nbr_fractions = np.ones(n_users)	
# 	x_ = np.random.uniform(low=0.0, high=xmax, size=n_users)
# 	y_ = np.random.uniform(low=0.0, high=ymax, size=n_users)
# 	xyUEs = np.vstack((x_,y_)) 
# 	xyBSs = np.vstack((250,250))
# 	# T_shard
# 	#CHANNEL GAIN
# 	Gains = channel_gain(n_users,1,1, xyUEs, xyBSs,0)
# 	gains = []
# 	for i in range(n_users):
# 		gains.append(Gains[0,i,0])

# 	cost = np.zeros(n_users)
# 	for k in range(len(L)):
# 		rmin = S/(T-Ttrain[k])
# 		c = 1
# 		r = 0
# 		print('required_bandwidth for', k)
# 		while(r<rmin and c<n_users):
# 			r = datarate(c/n_users, gains[i])
# 			c = c+1
# 		cost[k] = c
# 		print(cost[k])			
# 	return cost	

# def allocate(rep,cost,n_users):
# 	priority = [rep[i]/cost[i] for i in range(len(cost))]
# 	ordered = utils.get_order(priority)
# 	b = n_users 
# 	k = 0
# 	while(b>0):
# 		b = b-cost[k]
# 		k+=1
# 	return ordered[:k]	

def allocate_random(cost,n):
	priority = [1/cost[i] for i in range(len(cost))]
	ordered = utils.get_order(priority)
	b = n_users 
	k = 0
	while(b>0):
		b = b-cost[k]
		k+=1
	return ordered[:k]