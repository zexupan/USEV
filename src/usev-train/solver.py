import time
from utils import *
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.functional as F


class Solver(object):
    def __init__(self, train_data, validation_data, test_data, model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.args = args
        self.amp = amp

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter('logs/%s/tensorboard/' % args.log_name)

        self.model, self.optimizer = self.amp.initialize(model, optimizer,
                                                        opt_level=args.opt_level,
                                                        patch_torch_functions=args.patch_torch_functions)

        if self.args.distributed:
            self.model = DDP(self.model)

        self._reset()

    def _reset(self):
        self.halving = False
        if self.args.continue_from:
            checkpoint = torch.load('logs/%s/model_dict_last.pt' % self.args.continue_from, map_location='cpu')


            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.amp.load_state_dict(checkpoint['amp'])

            self.start_epoch=checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']

            if self.print: print("Resume training from epoch: {}".format(self.start_epoch))
            
        else:
            checkpoint = torch.load('../../pretrain_networks/sdr-av-dprnn-pretrain-noise.pt', map_location='cpu')
            pretrained_model=checkpoint['model']
            self.model.load_state_dict(pretrained_model)

            self.best_val_loss = float("inf")
            self.val_no_impv = 0
            self.start_epoch=1
            if self.print: print('Start new training')

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs+1):
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)
            # Train
            self.model.train()
            start = time.time()
            tr_loss, tr_avg_energy_1, tr_avg_sisnri_2, tr_avg_sisnri_3, tr_avg_energy_4 = self._run_one_epoch(data_loader = self.train_data)
            reduced_tr_loss = self._reduce_tensor(tr_loss)
            reduced_tr_avg_energy_1 = self._reduce_tensor(tr_avg_energy_1)
            reduced_tr_avg_sisnri_2 = self._reduce_tensor(tr_avg_sisnri_2)
            reduced_tr_avg_sisnri_3 = self._reduce_tensor(tr_avg_sisnri_3)
            reduced_tr_avg_energy_4 = self._reduce_tensor(tr_avg_energy_4)

            if self.print: print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Train Loss {2:.3f}'.format(
                        epoch, time.time() - start, reduced_tr_loss))

            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss, val_avg_energy_1, val_avg_sisnri_2, val_avg_sisnri_3, val_avg_energy_4 = self._run_one_epoch(data_loader = self.validation_data, state='val')
                reduced_val_loss = self._reduce_tensor(val_loss)
                reduced_val_avg_energy_1 = self._reduce_tensor(val_avg_energy_1)
                reduced_val_avg_sisnri_2 = self._reduce_tensor(val_avg_sisnri_2)
                reduced_val_avg_sisnri_3 = self._reduce_tensor(val_avg_sisnri_3)
                reduced_val_avg_energy_4 = self._reduce_tensor(val_avg_energy_4)

            if self.print: print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Valid Loss {2:.3f}'.format(
                          epoch, time.time() - start, reduced_val_loss))

            # test
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                test_loss, test_avg_energy_1, test_avg_sisnri_2, test_avg_sisnri_3, test_avg_energy_4 = self._run_one_epoch(data_loader = self.test_data, state='test')
                reduced_test_loss = self._reduce_tensor(test_loss)
                reduced_test_avg_energy_1 = self._reduce_tensor(test_avg_energy_1)
                reduced_test_avg_sisnri_2 = self._reduce_tensor(test_avg_sisnri_2)
                reduced_test_avg_sisnri_3 = self._reduce_tensor(test_avg_sisnri_3)
                reduced_test_avg_energy_4 = self._reduce_tensor(test_avg_energy_4)

            if self.print: print('Test Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Test Loss {2:.3f}'.format(
                          epoch, time.time() - start, reduced_test_loss))


            # Check whether to adjust learning rate and early stop
            find_best_model = False
            if reduced_val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = reduced_val_loss
                find_best_model=True

            # Halfing the learning rate
            self.halving = True
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] *0.98
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar('Train_loss', reduced_tr_loss, epoch)
                    self.writer.add_scalar('Validation_loss', reduced_val_loss, epoch)
                    self.writer.add_scalar('Test_loss', reduced_test_loss, epoch)

                    self.writer.add_scalar('Train_loss_1', reduced_tr_avg_energy_1, epoch)
                    self.writer.add_scalar('Validation_loss_1', reduced_val_avg_energy_1, epoch)
                    self.writer.add_scalar('Test_loss_1', reduced_test_avg_energy_1, epoch)

                    self.writer.add_scalar('Train_loss_2', reduced_tr_avg_sisnri_2, epoch)
                    self.writer.add_scalar('Validation_loss_2', reduced_val_avg_sisnri_2, epoch)
                    self.writer.add_scalar('Test_loss_2', reduced_test_avg_sisnri_2, epoch)

                    self.writer.add_scalar('Train_loss_3', reduced_tr_avg_sisnri_3, epoch)
                    self.writer.add_scalar('Validation_loss_3', reduced_val_avg_sisnri_3, epoch)
                    self.writer.add_scalar('Test_loss_3', reduced_test_avg_sisnri_3, epoch)

                    self.writer.add_scalar('Train_loss_4', reduced_tr_avg_energy_4, epoch)
                    self.writer.add_scalar('Validation_loss_4', reduced_val_avg_energy_4, epoch)
                    self.writer.add_scalar('Test_loss_4', reduced_test_avg_energy_4, epoch)

                # Save model
                checkpoint = {'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'amp': self.amp.state_dict(),
                                'epoch': epoch+1,
                                'best_val_loss': self.best_val_loss,
                                'val_no_impv': self.val_no_impv}
                torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_last.pt")
                if find_best_model:
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_best.pt")
                    print("Fund new best model, dict saved")

    def _run_one_epoch(self, data_loader, state='train'):
        total_loss = 0
        avg_sisnri_2 = []
        avg_sisnri_3 = []
        avg_energy_1 = []
        avg_energy_4 = []
        self.accu_count = 0
        self.optimizer.zero_grad()
        for i, (a_mix, a_tgt, v_tgt, label_tgt, label_int) in enumerate(data_loader):
            a_mix = a_mix.cuda().squeeze(0).float()
            a_tgt = a_tgt.cuda().squeeze(0).float()
            v_tgt = v_tgt.cuda().squeeze(0).float()
            label_tgt = label_tgt.cuda().squeeze(0).long()
            label_int = label_int.cuda().squeeze(0).long()

            a_est = self.model(a_mix, v_tgt)


            loss = 0
            for j in range(label_tgt.shape[0]):
                a_mix_utt, a_tgt_utt, a_est_utt = segment_utt(a_mix[j], a_tgt[j], a_est[j], label_tgt[j], label_int[j])
                energy_1, sisnr_2, sisnr_3, energy_4  = eval_segment_utt(a_mix_utt, a_tgt_utt,a_est_utt,state)

                if energy_1:
                    avg_energy_1.append(energy_1)
                    loss += energy_1*self.args.w_a
                if sisnr_2:
                    avg_sisnri_2.append(sisnr_2)
                    loss += sisnr_2*self.args.w_b
                if sisnr_3:
                    avg_sisnri_3.append(sisnr_3)
                    loss += sisnr_3*self.args.w_c
                if energy_4:
                    avg_energy_4.append(energy_4)
                    loss += energy_4*self.args.w_d
            loss = loss / label_tgt.shape[0]
            
            # print(loss.item())


            if state == 'train':
                self.accu_count += 1
                if self.args.accu_grad:
                    loss = loss/(self.args.effec_batch_size / self.args.batch_size)
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                                   self.args.max_norm)
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                                   self.args.max_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()            

            total_loss += loss.data

        tmp_value = loss*0+1.11

        if len(avg_energy_1) == 0: avg_energy_1 = tmp_value
        else: avg_energy_1 = sum(avg_energy_1)/len(avg_energy_1)

        if len(avg_sisnri_2) == 0: avg_sisnri_2 = tmp_value
        else: avg_sisnri_2 = sum(avg_sisnri_2)/len(avg_sisnri_2)

        if len(avg_sisnri_3) == 0: avg_sisnri_3 = tmp_value
        else: avg_sisnri_3 = sum(avg_sisnri_3)/len(avg_sisnri_3)

        if len(avg_energy_4) == 0: avg_energy_4 = tmp_value
        else: avg_energy_4 = sum(avg_energy_4)/len(avg_energy_4)

        total_loss = total_loss / (i+1)

        return total_loss, avg_energy_1, avg_sisnri_2, avg_sisnri_3, avg_energy_4

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt


def segment_utt(a_mix, a_tgt, a_est, label_tgt, label_int):
    utt_list = []
    for utt in [a_mix, a_tgt, a_est]:
        utt_1 = utt[((label_tgt == label_int) & (label_tgt == 0))]
        utt_3 = utt[((label_tgt == label_int) & (label_tgt == 1))]
        utt_2 = utt[((label_tgt != label_int) & (label_tgt == 1))]
        utt_4 = utt[((label_tgt != label_int) & (label_tgt == 0))]
        assert utt.shape[-1] == (utt_1.shape[-1]+ utt_2.shape[-1]+ utt_3.shape[-1]+ utt_4.shape[-1])
        utt_list.append([utt_1,utt_2,utt_3,utt_4]) 
    return utt_list[0], utt_list[1], utt_list[2]


def eval_segment_utt(a_mix, a_tgt,a_est,state):
    a_mix_1, a_mix_2, a_mix_3, a_mix_4= a_mix[0], a_mix[1], a_mix[2], a_mix[3]
    a_tgt_1, a_tgt_2, a_tgt_3, a_tgt_4= a_tgt[0], a_tgt[1], a_tgt[2], a_tgt[3]
    a_est_1, a_est_2, a_est_3, a_est_4= a_est[0], a_est[1], a_est[2], a_est[3]

    energy_1, sisnr_2, sisnr_3, energy_4 = None, None, None, None

    if a_mix_1.shape[-1]!=0:
        energy_1 = cal_logEnergy(a_est_1)

    if a_mix_2.shape[-1]!=0:
        sisnr_2 = - cal_SDR(a_tgt_2, a_est_2)

    if a_mix_3.shape[-1]!=0:
        sisnr_3 = - cal_SDR(a_tgt_3, a_est_3)

    if a_mix_4.shape[-1]!=0:
        energy_4 = cal_logEnergy(a_est_4)

    return energy_1, sisnr_2, sisnr_3, energy_4 

