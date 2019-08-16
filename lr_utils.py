from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import Callback

import os
import numpy as np
import warnings
import tensorflow.keras.backend as K

class LR_Finder(Callback):
    
    def __init__(self, start_lr=1e-5, end_lr=10, step_size=None, beta=.98):
        super().__init__()
        
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.step_size = step_size
        self.beta = beta
        self.lr_mult = (end_lr/start_lr)**(1/step_size)
        
    def on_train_begin(self, logs=None):
        self.best_loss = 1e9
        self.avg_loss = 0
        self.losses, self.smoothed_losses, self.lrs, self.iterations = [], [], [], []
        self.iteration = 0
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.start_lr)
        
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        self.iteration += 1
        
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smoothed_loss = self.avg_loss / (1 - self.beta**self.iteration)
        
        # Check if the loss is not exploding
        if self.iteration>1 and smoothed_loss > self.best_loss * 4:
            self.model.stop_training = True
            return

        if smoothed_loss < self.best_loss or self.iteration==1:
            self.best_loss = smoothed_loss
        
        lr = self.start_lr * (self.lr_mult**self.iteration)
        
        self.losses.append(loss)
        self.smoothed_losses.append(smoothed_loss)
        self.lrs.append(lr)
        self.iterations.append(self.iteration)
        
        
        K.set_value(self.model.optimizer.lr, lr)  
        
    def plot_lr(self):
        plt.xlabel('Iterations')
        plt.ylabel('Learning rate')
        plt.plot(self.iterations, self.lrs)
        
    def plot_loss(self, n_skip=10):
        plt.ylabel('Loss')
        plt.xlabel('Learning rate (log scale)')
        plt.plot(self.lrs[n_skip:-5], self.losses[n_skip:-5])
        plt.xscale('log')
        
    def plot_smoothed_loss(self, n_skip=10):
        plt.ylabel('Smoothed Losses')
        plt.xlabel('Learning rate (log scale)')
        plt.plot(self.lrs[n_skip:-5], self.smoothed_losses[n_skip:-5])
        plt.xscale('log')


# Code is ported from https://github.com/fastai/fastai
class OneCycleLR(Callback):
    def __init__(self,
                 max_lr,
                 min_lr,
                 sl_epoch=12,
                 end_epoch=4,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 batch_size=None,
                 steps_per_epoch=None,
                 verbose=True):
        """ This callback implements a cyclical learning rate policy (CLR).
        This is a special case of Cyclic Learning Rates, where we have only 1 cycle.
        After the completion of 1 cycle, the learning rate will decrease rapidly to
        100th its initial lowest value.

        # Arguments:
            sl_epoch: Int. Used for Slanted Triangular LR. It defines slant peak
            end_epoch: Int. Number of epochs for which steep reduction is done in the end
            maximum_momentum: Optional. Sets the maximum momentum (initial)
                value, which gradually drops to its lowest value in half-cycle,
                then gradually increases again to stay constant at this max value.
                Can only be used with SGD Optimizer.
            minimum_momentum: Optional. Sets the minimum momentum at the end of
                the half-cycle. Can only be used with SGD Optimizer.
            max_lr: Float. Max Learning Rate it will reach
            min_lr: Float. Min Learning Rate it will reach
            verbose: Bool. Whether to print the current learning rate after every
                epoch.

        # Reference
            - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)
            - [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
        """
        super(OneCycleLR, self).__init__()

        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.end_epoch = end_epoch
        self.max_momentum = maximum_momentum
        self.min_momentum = minimum_momentum
        self.verbose = verbose

        if self.max_momentum is not None and self.min_momentum is not None:
            self._update_momentum = True
        else:
            self._update_momentum = False

        self.clr_iterations = 0.
        self.history = {}

        self.epochs = None
        self.batch_size = batch_size
        self.steps = steps_per_epoch
        self.num_iterations = None
        self.sl_epoch = sl_epoch

    def _reset(self):
        """
        Reset the callback.
        """
        self.clr_iterations = 0.
        self.history = {}

    def compute_lr(self):
        """
        Compute the learning rate based on which phase of the cycle it is in.

        - If in the first half of training, the learning rate gradually increases.
        - If in the second half of training, the learning rate gradually decreases.
        - If in the final `end_percentage` portion of training, the learning rate
            is quickly reduced to near 100th of the original min learning rate.

        # Returns:
            the new learning rate
        """

        cycle_len = int(self.num_iterations - (self.end_epoch * self.steps))
        cycle_peak = self.sl_epoch * self.steps
        end_lr = (1./100.) * self.min_lr
        if self.clr_iterations > cycle_len:
            current_percentage = (self.clr_iterations - cycle_len) / (self.num_iterations - cycle_len)
            new_lr = self.min_lr - (current_percentage * (self.min_lr - end_lr))
        elif self.clr_iterations > cycle_peak:
            current_percentage = (self.clr_iterations - cycle_peak) / (cycle_len - cycle_peak)
            new_lr = self.max_lr - (current_percentage * (self.max_lr - self.min_lr))
        else:
            current_percentage = (self.clr_iterations) / (cycle_peak)
            new_lr = self.min_lr + (current_percentage * (self.max_lr - self.min_lr))

        if self.clr_iterations == self.num_iterations:
            self.clr_iterations = 0

        return new_lr

    def compute_momentum(self):
        """
         Compute the momentum based on which phase of the cycle it is in.

        - If in the first half of training, the momentum gradually decreases.
        - If in the second half of training, the momentum gradually increases.
        - If in the final `end_percentage` portion of training, the momentum value
            is kept constant at the maximum initial value.

        # Returns:
            the new momentum value
        """

        cycle_len = int(self.num_iterations - (self.end_epoch * self.steps))
        cycle_peak = self.sl_epoch * self.steps
        if self.clr_iterations > cycle_len:
            new_momentum = self.max_momentum
        elif self.clr_iterations > cycle_peak:
            current_percentage = (self.clr_iterations - cycle_peak) / (cycle_len - cycle_peak)
            new_momentum = self.min_momentum + (current_percentage * (self.max_momentum - self.min_momentum))
        else:
            current_percentage = (self.clr_iterations) / (cycle_peak)
            new_momentum = self.max_momentum - (current_percentage * (self.max_momentum - self.min_momentum))

        return new_momentum

    def on_train_begin(self, logs={}):
        logs = logs or {}

        self.epochs = self.params['epochs']
        if not self.batch_size:
            self.batch_size = self.params['batch_size']
        if not self.steps:
            self.steps = self.params['steps']

        if self.steps is not None:
            self.num_iterations = self.epochs * self.steps
        else:
            raise ValueError("steps is required")

        self._reset()
        K.set_value(self.model.optimizer.lr, self.compute_lr())

        if self._update_momentum:
            if not hasattr(self.model.optimizer, 'momentum'):
                raise ValueError("Momentum can be updated only on SGD optimizer !")

            new_momentum = self.compute_momentum()
            K.set_value(self.model.optimizer.momentum, new_momentum)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}

        self.clr_iterations += 1
        new_lr = self.compute_lr()

        self.history.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr))
        K.set_value(self.model.optimizer.lr, new_lr)

        if self._update_momentum:
            if not hasattr(self.model.optimizer, 'momentum'):
                raise ValueError("Momentum can be updated only on SGD optimizer !")

            new_momentum = self.compute_momentum()

            self.history.setdefault('momentum', []).append(
                K.get_value(self.model.optimizer.momentum))
            K.set_value(self.model.optimizer.momentum, new_momentum)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            if self._update_momentum:
                print(" - lr: %0.5f - momentum: %0.2f " %
                      (self.history['lr'][-1], self.history['momentum'][-1]))

            else:
                print(" - lr: %0.5f " % (self.history['lr'][-1]))

    def test_run(self, epochs=5):
        """
        Visualize values of learning rate (and momentum) as a function of iteration (batch).
        :param n_iter: a number of cycles. If None, 1000 is used.
        """

        if hasattr(self, 'clr_iterations'):
            original_it = self.clr_iterations

        self.num_iterations = epochs * self.steps
        
        lrs = np.zeros(shape=(self.num_iterations,))
        moms = np.zeros_like(lrs)

        for i in range(self.num_iterations):
            self.clr_iterations = i
            lrs[i] = self.compute_lr()
            moms[i] = self.compute_momentum()
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(lrs)
        plt.xlabel('iterations')
        plt.ylabel('lr')
        plt.subplot(1, 2, 2)
        plt.plot(moms)
        plt.xlabel('iterations')
        plt.ylabel('momentum')

        if hasattr(self, 'current_iter'):
            self.clr_iterations = original_it
