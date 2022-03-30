#!/usr/bin/env python

from __future__ import print_function
import time
import sys
from optparse import OptionParser
from common import *
from SyncNet.model import Model
from SyncNet.dataset import *
from common import DBType
from SyncNet.eval import Evaluator

parser = OptionParser()
parser.add_option('--train_config', type=str, help="training configuration", default="./SyncNet/config.yaml")
parser.add_option('--db_config', type=str, help="db configuration", default="./dbs_config.yaml")


def main(argv):
    os.chdir('..')

    (opts, args) = parser.parse_args(argv)
    config = ConfigParser(opts.train_config)
    db_config = ConfigParser(opts.db_config)

    # solves open-cv dead lock bug
    # multiprocessing.set_start_method('spawn', force=True)

    gpu_ids = np.array(config.general.gpu_ids.split(' ')).astype(np.int)
    if torch.cuda.is_available():
        device = torch.device('cuda:{0}'.format(gpu_ids[0]))
    else:
        device = torch.device('cpu')

    # torch.cuda.set_device(device)

    train_loader, dataset = init_dataset(config, db_config, DBType.Train)
    current_state_path = os.path.join(config.general.output_path, config.general.current_state_file_name)
    model = Model(config)

    if os.path.isfile(current_state_path):
        start_epoch, lr, not_improved_itr, last_train_loss = np.loadtxt(current_state_path, delimiter=',', dtype=float)
        start_epoch = int(start_epoch)
        not_improved_itr = int(not_improved_itr)
        model.update_learning_rate(lr)
        print('resuming from epoch %d' % start_epoch)
    else:
        start_epoch = 0
        not_improved_itr = 0
        last_train_loss = 99999999

    model.train()

    dataset_size = len(dataset)
    # logger = Logger(config)
    current_step = start_epoch * dataset_size

    evaluator = Evaluator(DBType.Validation, config, db_config)

    # if len(gpu_ids) > 1:
    # model = nn.DataParallel(model)

    model.to(device)
    tmp_loss = 0
    tmp_loss_count = 0
    tmp_time = time.time()

    for epoch in range(start_epoch, config.train.num_epochs):
        epoch_start_time = time.time()
        epoch_train_loss = 0

        for i, data in enumerate(train_loader):
            total_batches = current_step / config.train.batch_size
            epoch_iteration = current_step % dataset_size

            if total_batches % 3000 == 0:
                print('{} / {}'.format(epoch_iteration, dataset_size))

            visual_input = data['visual_input'].to(device)
            target_prediction = data['target_prediction'].to(device)
            audio_features = data['audio_features'].to(device)

            loss = model(visual_input, target_prediction, audio_features)
            loss = torch.mean(loss)

            model.optimizer.zero_grad()
            loss.backward()

            # if config.general.clip_grads:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            model.optimizer.step()

            epoch_train_loss += loss.item()
            tmp_loss += loss.item()
            tmp_loss_count += 1

            current_step += config.train.batch_size

            if tmp_loss_count % 3000 == 0:
                tmp_loss = tmp_loss / (tmp_loss_count)
                print('temp train loss {} | time = {}'.format(tmp_loss, time.time() - tmp_time))
                tmp_loss_count = 0
                tmp_loss = 0
                tmp_time = time.time()

        print('end of epoch %d / %d \t time taken: %d sec' %
              (epoch, config.train.num_epochs, time.time() - epoch_start_time))

        epoch_train_loss = epoch_train_loss / (i + 1)
        print('epoch training loss {}'.format(epoch_train_loss))

        losses_dict = {'epoch training loss': epoch_train_loss}
        # logger.dump_current_errors(losses_dict, current_step)

        model.save('latest')
        model.save(str(epoch))

        if epoch_train_loss > last_train_loss:
            not_improved_itr += 1

            if not_improved_itr == 1:
                model.update_learning_rate()
                not_improved_itr = 0
                last_train_loss = epoch_train_loss
        else:
            not_improved_itr = 0
            last_train_loss = epoch_train_loss

        if epoch % config.general.eval_epcohs_freq == 0:
            evaluator.eval(model, max_iterations=config.train.max_eval_iterations)

        np.savetxt(current_state_path, (epoch + 1, model.current_lr, not_improved_itr, last_train_loss), delimiter=',', fmt='%f')


if __name__ == '__main__':
    main(sys.argv)
