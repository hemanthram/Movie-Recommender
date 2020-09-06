import pandas as pd
import numpy as np
import random
import tensorflow as tf
import numpy as np
from pathlib import Path
from string import ascii_lowercase,digits
import sys


def find_com_words (name, words):
    count = 0
    blacklist = ['the','of','at','and','in','on'] + list(ascii_lowercase) + list(digits)
    for word in words:
        lower_word = word.lower()
        if (lower_word not in blacklist):
            if(name.find(word) != -1) :
                count = count + 1
    #print(count)
    if(count >= 2):
        return True
    else :
        return False

# Get and split input movies(liked) from command line
# Or look at the movie index and give input here

# print(len(sys.argv))
# sys.exit("jk")

if( len(sys.argv) == 1 ):
    input_movies = input("Enter the movie indices: ")
    input_movies = list(input_movies.split())
else:
    input_movies = sys.argv[1:]

input_movies = np.array(input_movies,dtype=int)

data = pd.read_csv('data/genres.csv')
genre_oh = np.array(data) # oh -> one hot encoding of particular genres

x = genre_oh[input_movies,:]
print(x)
m , n = x.shape

# Assuming that the user likes the input movies
y = np.ones((m,1))

print(m)
layer = 10
lr = 0.1
t,n = genre_oh.shape

# Beginning of calculations with tf
tf.reset_default_graph()
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
    ans = sess.run(test_ans, feed_dict={testx:genre_oh})

ans = np.reshape(np.array(ans),(ans.shape[0]))
order = np.argsort(ans)

movie_data = pd.read_csv('data/movie_metadata_edited.csv')
movie_names = movie_data['movie_title']
movie_genres = movie_data['genres']

# Append sequels and prequels for use later if needed
name_class = []
for test in movie_names[input_movies]:
    words = test.split()
    #print(words)
    i = 0
    for name in movie_names:
        if(find_com_words(name,words)):
            print(name,i)
            if(i not in input_movies):
                name_class.append(i)
        i = i + 1

name_class = np.array(name_class)

# Delete input movies from prediction
for i in input_movies:
    index = np.argwhere(order == i)
    order = np.delete(order,index) 

# Arrange in desc since we need movie index with highest score
order_final = np.flip(order)

# Make a dir for output if not there alerady
Path("output/").mkdir(parents=True, exist_ok=True)
print(name_class)

# Ouput our predictions as JSON
# names.json -> movies with similar names(sequels)
# predicted_movies -> Top 20 predicted movies
movie_data.iloc[name_class].to_json('output/names.json','records')
movie_data.iloc[order_final[0:20]].to_json('output/predicted_movies.json','records')