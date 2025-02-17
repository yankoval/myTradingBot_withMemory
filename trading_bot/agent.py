import random

from collections import deque

import numpy as np
import tensorflow as tf
import os


import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Dense
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam

def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        print("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """
    ver = 'v01'
    description = 'Base q-lerning agen and model'

    def __init__(self, state_size, strategy="t-dqn", reset_every=1000
                 , pretrained=False, model_name=None):

        print('myTradingBot_withMemory')

        # Switch off gpu if not detected
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # Check strategy
        if strategy not in ["t-dqn", "double-dqn", "dqn"]:
            raise
        self.strategy = strategy

        # agent config
        self.state_size = state_size    	# normalized previous days
        self.action_size = 3           		# [sit, buy, sell]
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 0.99
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0005
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(learning_rate=self.learning_rate)

        if pretrained:
            self.model = self.load()
            config = self.model.get_config()  # Returns pretty much every information about your model
            self.state_size = config["layers"][0]["config"]["batch_input_shape"][1]
            print(f'Loaded model at:{self.model_name}, window size:{self.state_size}.')
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Creates the model
        """
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        if type(action) not in [np.int64,int]:
            raise 
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size),None

        # if self.first_iter:
        #     self.first_iter = False
        #     return 1 # make a definite buy on the first iter

        action_probs = self.model.predict(state,verbose=0)
        action = np.argmax(action_probs[0])
        if type(action) not in [np.int64,int]:
            raise
        return action, action_probs

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        
        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation
                    target = reward + self.gamma * np.amax(self.model.predict(next_state,verbose=0)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state,verbose=0)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation with fixed targets
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state,verbose=0)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state,verbose=0)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # Double DQN
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights(),verbose=0)

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate double deep q-learning equation
                    target = reward + self.gamma * self.target_model.predict(next_state,verbose=0)[0][np.argmax(self.model.predict(next_state,verbose=0)[0])]

                # estimate q-values based on current state
                q_values = self.model.predict(state,verbose=0)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])
                
        else:
            raise NotImplementedError()

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save(f"models/{self.model_name}_episode_{episode}")

    def load(self):
        try:
            return load_model("models/" + self.model_name, custom_objects=self.custom_objects)
        except:
            print(f'Error reading model from:{os.getcwd()} and {"models/" + self.model_name}')
            raise


class AgentF(Agent):
    ver = 'v02'
    description = 'Base q-lerning agen and model for fractals seq relu 6 layers'

    def _model(self):
        """Creates the model
        """
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model
    
    