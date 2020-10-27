
# SRGAN x Optuna

import argparse as arg
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import tensorflow.keras.layers as kl
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import backend as K

import cv2
import numpy as np
import matplotlib.pyplot as plt

import optuna
import uuid

# Super-resolution Image Generator
class Generator(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()

        input_shape_ps = (input_shape[0], input_shape[1], 64) 

        # Pre stage(Down Sampling)
        self.pre = [
            kl.Conv2D(64, kernel_size=9, strides=1,
                    padding="same", input_shape=input_shape),
            kl.Activation(tf.nn.relu)
        ]

        # Residual Block
        self.res = [
            [
                Res_block(64, input_shape) for _ in range(7)
            ]
        ]

        # Middle stage
        self.middle = [
            kl.Conv2D(64, kernel_size=3, strides=1, padding="same"),
            kl.BatchNormalization()
        ]

        # Pixel Shuffle(Up Sampling)
        self.ps =[
            [
                Pixel_shuffler(128, input_shape_ps) for _ in range(2)
            ],
            kl.Conv2D(3, kernel_size=9, strides=4, padding="same", activation="tanh")
        ]

    def call(self, x):

        # Pre stage
        pre = x
        for layer in self.pre:
            pre = layer(pre)

        # Residual Block
        res = pre
        for layer in self.res:
            for l in layer:
                res = l(res)
        
        # Middle stage
        middle = res
        for layer in self.middle:
            middle = layer(middle)
        middle += pre

        # Pixel Shuffle
        out = middle
        for layer in self.ps:
            if isinstance(layer, list):
                for l in layer:
                    out = l(out)
            else:
                out = layer(out)

        return out

# Discriminator 
class Discriminator(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = kl.Conv2D(64, kernel_size=3, strides=1,
                            padding="same", input_shape=input_shape)
        self.act1 = kl.Activation(tf.nn.relu)

        self.conv2 = kl.Conv2D(64, kernel_size=3, strides=2,
                            padding="same")
        self.bn1 = kl.BatchNormalization()
        self.act2 = kl.LeakyReLU()

        self.conv3 = kl.Conv2D(128, kernel_size=3, strides=1,
                            padding="same")
        self.bn2 = kl.BatchNormalization()
        self.act3 = kl.LeakyReLU()

        self.conv4 = kl.Conv2D(128, kernel_size=3, strides=2,
                            padding="same")
        self.bn3 = kl.BatchNormalization()
        self.act4 = kl.LeakyReLU()

        self.conv5 = kl.Conv2D(256, kernel_size=3, strides=1,
                            padding="same")
        self.bn4 = kl.BatchNormalization()
        self.act5 = kl.LeakyReLU()

        self.conv6 = kl.Conv2D(256, kernel_size=3, strides=2,
                            padding="same")
        self.bn5 = kl.BatchNormalization()
        self.act6 = kl.LeakyReLU()

        self.conv7 = kl.Conv2D(512, kernel_size=3, strides=1,
                            padding="same")
        self.bn6 = kl.BatchNormalization()
        self.act7 = kl.LeakyReLU()

        self.conv8 = kl.Conv2D(512, kernel_size=3, strides=2,
                            padding="same")
        self.bn7 = kl.BatchNormalization()
        self.act8 = kl.LeakyReLU()

        self.flt = kl.Flatten()

        self.dens1 = kl.Dense(1024, activation=kl.LeakyReLU())
        self.dens2 = kl.Dense(1, activation="sigmoid")

    def call(self, x):

        d1 = self.act1(self.conv1(x))
        d2 = self.act2(self.bn1(self.conv2(d1)))
        d3 = self.act3(self.bn2(self.conv3(d2)))
        d4 = self.act4(self.bn3(self.conv4(d3)))
        d5 = self.act5(self.bn4(self.conv5(d4)))
        d6 = self.act6(self.bn5(self.conv6(d5)))
        d7 = self.act7(self.bn6(self.conv7(d6)))
        d8 = self.act8(self.bn7(self.conv8(d7)))

        d9 = self.dens1(self.flt(d8))
        d10 = self.dens2(d9)

        return d10

# Pixel Shuffle
class Pixel_shuffler(tf.keras.Model):
    def __init__(self, out_ch, input_shape):
        super().__init__()

        self.conv = kl.Conv2D(out_ch, kernel_size=3, strides=1,
                            padding="same", input_shape=input_shape)
        self.act = kl.Activation(tf.nn.relu)
    
    # forward proc
    def call(self, x):

        d1 = self.conv(x)
        d2 = self.act(tf.nn.depth_to_space(d1, 2))

        return d2

# Residual Block
class Res_block(tf.keras.Model):
    def __init__(self, ch, input_shape):
        super().__init__()

        self.conv1 = kl.Conv2D(ch, kernel_size=3, strides=1,
                            padding="same", input_shape=input_shape)
        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)

        self.conv2 = kl.Conv2D(ch, kernel_size=3, strides=1,
                            padding="same")
        self.bn2 = kl.BatchNormalization()

        self.add = kl.Add()

    def call(self, x):

        d1 = self.av1(self.bn1(self.conv1(x)))
        d2 = self.bn2(self.conv2(d1))

        return self.add([x, d2])

# Train
class trainer():
    def __init__(self, lr_img, hr_img, out_path, batch_size):

        lr_shape = lr_img.shape # Low-resolution Image shape
        hr_shape = hr_img.shape # High-resolution Image shape

        # Content Loss Model setup
        input_tensor = tf.keras.Input(shape=hr_shape)
        self.vgg = VGG16(include_top=False, input_tensor=input_tensor)
        self.vgg.trainable = False
        self.vgg.outputs = [self.vgg.layers[9].output]  # VGG16 block3_conv3 output  

        # Content Loss Model
        self.cl_model = tf.keras.Model(input_tensor, self.vgg.outputs)

        # Discriminator
        discriminator_ = Discriminator(hr_shape)
        inputs = tf.keras.Input(shape=hr_shape)
        outputs = discriminator_(inputs)
        self.discriminator = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Generator
        self.generator = Generator(lr_shape)
        
        # Combined Model setup
        lr_input = tf.keras.Input(shape=lr_shape)
        sr_output = self.generator(lr_input)

        self.discriminator.trainable = False # Discriminator train Disable
        d_fake = self.discriminator(sr_output)

        # SRGAN Model
        self.gan = tf.keras.Model(inputs=lr_input, outputs=[sr_output, d_fake])

        self.weight_path = os.path.join(out_path, "srgan") # Parameter-file output path
        self.graph_path = os.path.join(out_path, "graph.jpg") # Graph-image output path

        self.h_batch = int(batch_size / 2) # Half mini batch size

        self.real_lab = np.ones((self.h_batch, 1))  # High-resolution image label
        self.fake_lab = np.zeros((self.h_batch, 1)) # Super-resolution image label(Discriminator side)
        self.gan_lab = np.ones((self.h_batch, 1))

    # Content loss
    def Content_loss(self, hr_img, sr_img):
        return K.mean(K.abs(K.square(self.cl_model(hr_img) - self.cl_model(sr_img))))

    # PSNR
    def psnr(self, hr_img, sr_img):
        return cv2.PSNR(hr_img, sr_img)
        
    def train(self, trial, lr_imgs, hr_imgs, epoch, optuna_ED=False):

        trial_uuid = str(uuid.uuid4())
        trial.set_user_attr("uuid", trial_uuid)

        # Optuna Enable
        if optuna_ED:
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2) # Adam Learning-rate
            loss_weight_a = trial.suggest_loguniform('loss_weight_a', 1e-1, 1.5) # content-loss weight
            loss_weight_b = trial.suggest_loguniform('loss_weight_b', 1e-5, 1e-2) # BCE weight
        else: # Optuna Disable
            learning_rate = 0.001
            loss_weight_a = 1
            loss_weight_b = 1e-3

        # Discriminator
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                loss=tf.keras.losses.BinaryCrossentropy(),
                                metrics=['accuracy'])

        # SRGAN
        self.gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss=[self.Content_loss, tf.keras.losses.BinaryCrossentropy()],
                        loss_weights=[loss_weight_a, loss_weight_b])

        g_loss_plt = [] # Generator Loss list

        print("")
        
        # train run
        print("UUID: ", trial_uuid)
        for epoch in range(epoch):

            # - Train Discriminator -

            # High-resolution image random pickups
            idx = np.random.randint(0, hr_imgs.shape[0], self.h_batch)
            hr_img = hr_imgs[idx]

            # Low-resolution image random pickups
            lr_img = lr_imgs[idx]

            # Discriminator enabled train
            self.discriminator.trainable = True

            # train by High-resolution image
            d_real_loss = self.discriminator.train_on_batch(hr_img, self.real_lab)

            # train by Super-resolution image
            sr_img = self.generator.predict(lr_img) 
            d_fake_loss = self.discriminator.train_on_batch(sr_img, self.fake_lab)
            
            # Discriminator average loss 
            d_loss = 0.5 * np.add(d_real_loss, d_fake_loss)
            
            # - Train Generator -

            # High-resolution image random pickups
            idx = np.random.randint(0, hr_imgs.shape[0], self.h_batch)
            hr_img = hr_imgs[idx]

            # Low-resolution image random pickups
            lr_img = lr_imgs[idx]

            # train by Generator
            self.discriminator.trainable = False
            g_loss = self.gan.train_on_batch(lr_img, [hr_img, self.gan_lab])
            sr_img = self.generator.predict(lr_img) 

            # Epoch num, Discriminator/Generator loss, PSNR
            PSNR = self.psnr(hr_img, sr_img)
            print("\rEpoch: {0} D_loss: {1:.3f} G_loss: {2:.3f} PSNR: {3:.3f}".format(epoch+1, d_loss[0], g_loss[0], PSNR), end="")

            g_loss_plt.append(g_loss[0])

            obj_func = g_loss[0] # SRGAN Loss
            trial.report(obj_func, epoch+1)
            # Pruning
            if trial.should_prune():
                print("\nCeased at the {} learning".format(epoch+1))
                raise optuna.exceptions.TrialPruned()

            # Plotting and Saving the loss value
            if (epoch+1) % 10 == 0:
                plt.plot(g_loss_plt)
                plt.savefig(self.graph_path, bbox_inches='tight', pad_inches=0.1)        

        # Parameter-File and Graph Saving
        print("\n___Saving parameter...", end="")
        self.generator.save_weights(self.weight_path+"_"+trial_uuid+".h5") # Parameter-File Saving

        plt.plot(g_loss_plt, label=trial_uuid[:4])
        plt.legend()
        plt.savefig(self.graph_path, bbox_inches='tight', pad_inches=0.1) # Graph Saving
        print("\r___Saving parameter...Successfully completed")

        return obj_func

# Dataset creation
def create_dataset(data_dir, h, w, mag):

    print("\n___Creating a dataset...")
    
    prc = ['/', '-', '\\', '|']
    cnt = 0

    print("Number of image in a directory: {}".format(len(os.listdir(data_dir))))

    lr_imgs = []
    hr_imgs = []
    for c in os.listdir(data_dir):
        d = os.path.join(data_dir, c)

        _, ext = os.path.splitext(c)
        if ext.lower() not in ['.jpg', '.png', '.bmp']:
            continue

        img = cv2.imread(d)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (h, w)) # High-resolution image

        img_low = cv2.resize(img, (int(h/mag), int(w/mag))) # Image reduction
        img_low = cv2.resize(img_low, (h, w)) # Resize to original size

        lr_imgs.append(img_low)
        hr_imgs.append(img)

        cnt += 1

        print("\rLoading a LR-images and HR-images...{}    ({} / {})".format(prc[cnt%4], cnt, len(os.listdir(data_dir))), end='')

    print("\rLoading a LR-images and HR-images...Done    ({} / {})".format(cnt, len(os.listdir(data_dir))), end='')

    # Low-resolution image
    lr_imgs = tf.convert_to_tensor(lr_imgs, np.float32) 
    lr_imgs = (lr_imgs.numpy() - 127.5) / 127.5

    # High-resolution image
    hr_imgs = tf.convert_to_tensor(hr_imgs, np.float32)
    hr_imgs = (hr_imgs.numpy() - 127.5) / 127.5
    
    print("\n___Successfully completed\n")

    return lr_imgs, hr_imgs

def main():

    # Command line option
    parser = arg.ArgumentParser(description='Super-resolution GAN training')
    parser.add_argument('--data_dir', '-d', type=str, default=None,
                        help='Specify the image folder path (If not specified, an error)')
    parser.add_argument('--out', '-o', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Specify where to save parameters (default: ./srgan_xxx.h5)')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='Specify the mini-batch size (default: 32)')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Specify the number of times to train (default: 1000)')
    parser.add_argument('--he', '-he', type=int, default=128,
                        help='Resize height (default: 128)')      
    parser.add_argument('--wi', '-wi', type=int, default=128,
                        help='Resize width (default: 128)')
    parser.add_argument('--mag', '-m', type=int, default=2,
                        help='Magnification (default: 2)')
    parser.add_argument('--optuna_ED', '-op', action='store_true', 
                        help='Optuna Enable/Disable')  
    parser.add_argument('--optuna_num', '-opn', type=int, default=10,
                        help='HP-Optimization Trials (default: 10)')  
    args = parser.parse_args()

    # Image folder not specified. -> Exception
    if args.data_dir == None:
        print("\nException: Folder not specified.\n")
        sys.exit()
    # An image folder that does not exist was specified. -> Exception
    if os.path.exists(args.data_dir) != True:
        print("\nException: Folder \"{}\" is not found.\n".format(args.data_dir))
        sys.exit()
    # When 0 is entered for either width/height or Reduction ratio. -> Exception
    if args.he == 0 or args.wi == 0 or args.mag == 0:
        print("\nException: Invalid value has been entered.\n")
        sys.exit()

    # Create output folder (If the folder exists, it will not be created.)
    os.makedirs(args.out, exist_ok=True)

    # Setting info
    print("=== Setting information ===")
    print("# Images folder: {}".format(os.path.abspath(args.data_dir)))
    print("# Output folder: {}".format(args.out))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# Epoch: {}".format(args.epoch))
    print("")
    print("# Height: {}".format(args.he))
    print("# Width: {}".format(args.wi))
    print("# Magnification: {}".format(args.mag))
    print("")
    print("# Optuna: {}".format(args.optuna_ED))
    print("# HP-Optimization Trials: {}".format(args.optuna_num))
    print("===========================")

    # dataset creation
    lr_imgs, hr_imgs = create_dataset(args.data_dir, args.he, args.wi, args.mag)
    
    print("___Start training...")
    
    if args.optuna_ED:
        def objective(trial):
            Trainer = trainer(lr_imgs[0], hr_imgs[0], out_path=args.out, batch_size=args.batch_size)
            obj_func = Trainer.train(trial, lr_imgs, hr_imgs, epoch=args.epoch, optuna_ED=args.optuna_ED)

            return obj_func

        plt.figure(figsize=(12.8, 8.0), dpi=100)
        Trainer_op = optuna.create_study()
        Trainer_op.optimize(objective, n_trials=args.optuna_num)
        print("___Training finished\n\n")

        print("Best UUID: ", Trainer_op.best_trial.user_attrs['uuid'])
        print("Best params: ", Trainer_op.best_params)
        print("Best test accuracy: ", Trainer_op.best_value)
    else:
        Trainer = trainer(lr_imgs[0], hr_imgs[0])
        Trainer.train(lr_imgs, hr_imgs, out_path=args.out, batch_size=args.batch_size, epoch=args.epoch)

if __name__ == '__main__':
    main()