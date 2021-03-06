"""
train our small qa system
"""

import argparse
import os
import tensorflow as tf
import sys
from data import QALoader
from qa_model import AttentionQA
from model import PointerNet
from tqdm import tqdm


def train(args):
    # load data
    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    training = QALoader(os.path.join(args.data_dir, 'train.txt'), vocab_path, args.batch_size, 45, 6)
    validation = QALoader(os.path.join(args.data_dir, 'validate.txt'), vocab_path, args.batch_size, 45, 6)

    # create TensorFlow graph
    qa_net = AttentionQA(batch_size=args.batch_size, learning_rate=args.learning_rate)
    saver = tf.train.Saver()
    best_val_acc = 0
    best_loss = 100

    # record training loss & accuracy
    train_losses = []
    train_accuracies = []

    # initialize graph
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(args.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(args.log_dir + '/val')
        sess.run(init)
        # saver.restore(sess, os.path.join(args.save_dir, '201804030240','ptr_net.ckpt'))
        # saver.restore(sess, os.path.join(args.save_dir, '201804060454', 'ptr_net.ckpt'))
        # print('restored 98% model')
        for ep in tqdm(range(args.n_epochs)):
            tr_loss, tr_acc = 0, 0
            for itr in tqdm(range(training.n_batches)):
                x_batch, x_lengths, q_batch, q_length, y_batch = training.next_batch()
                train_dict = {qa_net.encoder_inputs: x_batch,
                              qa_net.input_lengths: x_lengths,
                              qa_net.question_inputs: q_batch,
                              qa_net.question_lengths: q_length,
                              qa_net.pointer_labels: y_batch}
                align, loss, acc, _ = sess.run([qa_net.qa_alignment, qa_net.loss, qa_net.exact_match, qa_net.train_step], feed_dict=train_dict)
                tr_loss += loss
                tr_acc += acc

            train_losses.append(tr_loss / training.n_batches)
            train_accuracies.append(tr_acc / training.n_batches)

            # tensorboard
            summ = sess.run(merged, feed_dict=train_dict)
            train_writer.add_summary(summ, ep)

            # check validation accuracy every 10 epochs
            if ep % 10 == 0:

                val_acc = 0
                for itr in range(validation.n_batches):
                    x_batch, x_lengths, q_batch, q_length, y_batch = validation.next_batch()
                    val_dict = {qa_net.encoder_inputs: x_batch,
                                qa_net.input_lengths: x_lengths,
                                qa_net.question_inputs: q_batch,
                                qa_net.question_lengths: q_length,
                                qa_net.pointer_labels: y_batch}
                    val_acc += sess.run(qa_net.exact_match, feed_dict=val_dict)
                val_acc = val_acc / validation.n_batches
                print('epoch {:3d}, loss={:.3f}'.format(ep, tr_loss / training.n_batches))
                print('Train EM: {:.3f}, Validation EM: {:.3f}'.format(tr_acc / training.n_batches, val_acc))

                summ = sess.run(merged, feed_dict=val_dict)
                test_writer.add_summary(summ, ep)

                # training.shuffle_batch()
                # save model
                if val_acc > best_val_acc:
                    print('Validation accuracy increased. Saving model.')
                    saver.save(sess, os.path.join(args.save_dir, 'ptr_net.ckpt'))
                    best_val_acc = val_acc
                else:
                    print('Validation accuracy decreased. Restoring model.')
                    saver.restore(sess, os.path.join(args.save_dir, 'ptr_net.ckpt'))
                    training.shuffle_batch()

        print('Training complete.')
        print('Best Validation EM: {:.2f}'.format(best_val_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory in which data is stored.')
    parser.add_argument('--save_dir', type=str, default='./models', help='Where to save checkpoint models.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Where to save checkpoint models.')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs to run.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for Adam optimizer.')
    args = parser.parse_args(sys.argv[1:])
    train(args)
