# copyright Q_NetLearning Clean 
#https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0


import gym 
import numpy as numpy 
import random 
import tensorflow as tf 
import matplotlib.pyplot as plt 


env=gym.make('FrozenLake-vO')


tf.reset_default_graph()

## 1-Feedfoward network 
# the state encoded in a one-hot vector (1x16)
input1=tf.placeholder(shape=[1,16],dtype=tf.float32)
#produces a vector of 4 Q-values
W=tf.Variable(tf.random_uniform([16,4],0,0.01))
# Prediction Q values 
Qout=tf.matmult(inputs1,W)
# Highest probability =action 
predict=tf.argmax(Qout,1)

## 2-Obtain the loss by taking the sum of square difference 
# between the target and prediction Q values.nextQ is the target Q_values 
nextQ=tf.placeholder(shape=[1,4],dtype=tf.float32)
# nextQ is the TargetQ calculated with the Bellman Equation with the Q value of the state s1 
#(output of the network when the input is state 1)
# Q out is the predict Q value 
loss=tf.reduce_sum(tf.mean(nextQ-Qout))
trainer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel=trainer.minimize(loss)

#3-Training the network 

init=tf.initialize_all_Variables()

##learning parameter
# discount variable 
y=0.99 
# greedy variable 
e=0.1 
#number of sessions (epochs)
num_episodes=2000

##create lists to contain total rewards and step per episods 
jList=[] # steps
rList=[] # rewards 

## create the tensorflow Session 
with tf.Sessioin as sess():
	sess.run(init)
	
	# iterate across the epochs 
	for i in range(num_episodes):
		#reset the environment and get the first new observation 
		s=env.reset()
		#rAll to feed the list of rewards 
		rAll=0
		# d = done 
		#done (boolean): whether it's time to reset the environment again. 
		#Most (but not all)                                                                                           tasks are divided up into well-defined episodes,
		#and done being True indicates the episode has terminated. 
		#(For example, perhaps the pole tipped too far, or you lost your last life.)
		d=False
		# j is the number of steps 
		j=0 

		# the Q-Table learning algorithm 
		 while j< 99 :
		 	j+=1 
		 	
		 	#1_Choose and action
		 	# Choose an action by greedy (with noise) picking from the Q network 
		 	# a is the action predict=tf.argmax(Qout,1)
		 	# allQ is all the Q_values possinle when s is the input state. Qout=tf.matmult(inputs1,W)
		 	a,allQ =sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
		 	if np.random.rand(1) < e:
		 		# environment after the action 
                a[0] = env.action_space.sample() 
		 	
		 	
		 	# 2_After this first action a[O], update the state (s1) and reward (r) from the environment 
		 	 s1,r,d,_ = env.step(a[0])

		 	
		 	 # 3_Obtain all the Q_values possible when s1 is the input state. 
		 	 # Obtain the Q1 values by feeding the new state s1 in the network 
		 	 # Q1 values = Qout values if the state S1 is the input1 of the network 
		 	 Q1=sess.run(Qout,feed_dict={inputs1: np.identity(16)[s1:s1+1]})

		 	
		 	 # Apply Belman Equation Q(s,a) = r + γ(max(Q(s’,a’))
		 	 # Obtain maxQ1 and set our target value for the choosen action at state s 
		 	 maxQ1=np.max(Q1) # ~ maxQ(s',a')
		 	 targetQ=allQ # targetQ[0,a[Ø]] ~  Q(s,a)
		 	 targetQ[0,a[Ø]]=r+y*maxQ1 # Q(s,a) = r + γ(max(Q(s’,a’))

		 	 
		 	 # Train our network using target and predicted Q values 
		 	 # nextQ is targetQ needed to train the model loss=tf.reduce_sum(tf.mean(nextQ-Qout))
		 	 _,W1=sess.run([updateModel,W], feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ}})

		 	 rAll += r
             s = s1
             if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
		print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"
	



