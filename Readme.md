
# Import the MNIST Dataset
(MNIST is a subset of the NIST dataset of handwritten numbers)


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz


# Import Tensorflow


```python
import tensorflow as tf
```

# Set up variables
$x$ will be a placeholder for values that tensorflow will use to run computations.

$W$ is a modifiable tensor (tf.variable) that will contain the weignts of the learned computations. The shape is 748x10 so it matches the shape of a flattened 28x28 matrix from the input augmented with a 'one-hot' vector. Since it's unknown what values $W$ will contain, it's set to zeros.

$b$ is a modifiable tensor (tf.Variable) that will contain the biases for the learned computation. Like $W$, it's unknown what will be in $b$ at this point so set it to zeros.


```python
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

# Set up the regression model

Below are two types of regression, softmax and logistic regressions. 

**Softmax** is a regression model based on the function: $$softmax(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}$$

**Logistic** is expressed as the sigmoid function: $$sigmoid(x_i) = \frac{1}{1+e^{-x_i}}$$

Experimenting around with these two functions leads me to believe that the softmax function is a more accurate model.


```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
#y = tf.nn.sigmoid(tf.matmul(x, W) + b)
```

# Train the model

## Create a placeholder for the correct values


```python
y_ = tf.placeholder(tf.float32, [None, 10])
```

## Compute the Cost

Below are two different cost functions that can be used to train data; Cross Entropy, and Squared Error.

**Cross Entropy** is defined as: $$H_{y'}(y) = \sum_{i} {y}_{i}^{'}log_2(y_i)$$

where $y$ is the *predicted* probability distribution and $y'$ is the true distribution given in the one-hot vector label.

**Squared Error** in turn is expressed as: $$ \frac{1}{2} \sum_{i}(y_{i}^{'}-y_{i})$$


```python
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
square_error = tf.reduce_mean(0.5*tf.square(y_ - y))
```

## Prepare the Training actions

Think of gradient descent as a hill sliding algorithm. That's to say that gradient descent will always try to move toward a local minima of a given function by the steepest route on every iteration.


```python
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```

### Initialize all of the Tensorflow variables


```python
init = tf.initialize_all_variables()
```

### Launch the model in a Session, and run the operation that initializes the variables.


```python
sess = tf.Session()
sess.run(init)
```

## Run the training step 

This runs the training step through the gradient descent 1000 times, each time getting closer to what it predicts the digit should be.


```python
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

# Evaluate the Model


```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```


```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

## Print the Accuracy of the Predictions


```python
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

    0.9134

