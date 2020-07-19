import tensorflow as tf

x = tf.random.uniform([1000,200,200])
y = tf.random.uniform([1000,200,200])

print('Start')

for _ in range(1000000):
    z = x*y

print('Finished')