import pandas as pd
import numpy as np
import random
import tensorflow as tf
import numpy as np
import sys


def find_com_words (name, words):
    count = 0
    for word in words:
        if (word not in ['The','the','of','Of','at','At','a','Z','and','And','A','in','In','on','On']):
            if(name.find(word)!=-1) :
                count = count +1
    #print(count)
    if(count >=2):
        return True
    else :
        return False

inpz = sys.argv[1]
#print(list(inpz))
n = list(inpz.split())

for i in ['\'']:
 inpz = inpz.replace(i,'')
n = inpz.split()
for i in range(len(n)):
 n[i] = n[i].replace('\'','')
 n[i] = int(n[i])
movies = np.array(n)

dat = pd.read_csv('genres .csv')
tot = np.array(dat)

#movies = np.array([41,38,94,72,29,52,91,39,63,24,31,49])
#x = np.array(dat)
x = tot[movies,:]
m , n = x.shape
y = np.ones((m,1))

tf.reset_default_graph()
print(m)
layer = 10
lr = 0.1
t,n = tot.shape

X = tf.placeholder(tf.float32,(m,n))
testx = tf.placeholder(tf.float32,(t,n))
testy = tf.get_variable("ty",shape=(t,1))
Y = tf.placeholder(tf.float32,(m,1))
w1 = tf.get_variable("w1",shape=(n,layer),initializer=tf.contrib.layers.xavier_initializer(seed=0))
w2 = tf.get_variable("w2",shape=(layer,1),initializer=tf.contrib.layers.xavier_initializer(seed=0))

l = (X@w1)
res = l@w2

l1 = (testx@w1)
test_ans = tf.sigmoid(l1@w2)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=res, labels=Y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, cur_cost = sess.run([optimizer, cost], feed_dict = {X:x, Y:y})
        #print(i+1,cur_cost)
    #cur_cost = sess.run(cost, feed_dict={X:x, Y:y})
    #print(cur_cost)
    ans = sess.run(test_ans, feed_dict={testx:tot})
ans = np.reshape(np.array(ans),(ans.shape[0]))

m_dat = pd.read_csv('movie_metadata_edited.csv')
m_names = m_dat['movie_title']
m_genres = m_dat['genres']
#print(m_names[movies],m_genres[movies])

order = np.argsort(ans)
#print(m_names[order[950:]],m_genres[order[950:]])

name_class = []
for test in m_names[movies]:
    words = test.split()
    #print(words)
    i = 0
    for name in m_names:
        if(find_com_words(name,words)):
            # print(name,i)
            if(i not in movies):
                name_class.append(i)
        i = i+1

name_class = np.array(name_class)

to_delete = []
for i in range(len(order)) :
    if(order[i] in movies):
        to_delete.append(i)
to_delete = np.array(to_delete)
order_f= np.delete(order,to_delete)
order_f = np.flip(order_f)

print(name_class)
m_dat.iloc[name_class].to_json('names.json','records')
m_dat.iloc[order_f[0:20]].to_json('genres.json','records')