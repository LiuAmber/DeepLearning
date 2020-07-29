from LoadData import LoadData
import model
import loss
import tensorflow as tf
import time
import matplotlib.pyplot as plt
EPOCHS = 150
import datetime
import os
from IPython import display

class train:
    def __init__(self):
        self.log_dir = "logs/"
        self.summary_writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.generator = model.Generator()
        self.discriminator = model.Discriminator()
        self.generator_optimizer = loss.generator_optimizer()
        self.discriminator_optimizer = loss.discriminator_optimizer()

        self.checkpoint,self.checkpoint_prefix = self.__get_checkpoint__()

    def __get_checkpoint__(self):
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        return checkpoint,checkpoint_prefix

    def __generate_images__(self,model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    @tf.function
    def __train_step__(self,input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = loss.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = loss.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

    def __fit__(self,train_ds, epochs, test_ds):
        for epoch in range(epochs):
            start = time.time()

            display.clear_output(wait=True)

            for example_input, example_target in test_ds.take(1):
                self.__generate_images__(self.generator, example_input, example_target)
            print("Epoch: ", epoch)

            # Train
            for n, (input_image, target) in train_ds.enumerate():
                print('.', end='')
                if (n + 1) % 100 == 0:
                    print()
                self.__train_step__(input_image, target, epoch)
            print()

            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def train(self):
        train_dataset = LoadData().train_data()
        test_dataset = LoadData().test_data()
        self.__fit__(train_dataset, EPOCHS, test_dataset)

train().train()





