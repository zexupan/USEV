import time
from utils import *

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.functional as F

class Solver(object):
    def __init__(self, train_data, validation_data, test_data, model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.args = args

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter('logs/%s/tensorboard/' % args.log_name)

        if self.args.distributed:
            from apex import amp
            from apex.parallel import DistributedDataParallel as DDP
            self.amp = amp
            self.model, self.optimizer = self.amp.initialize(model, optimizer, opt_level=args.opt_level, patch_torch_functions=args.patch_torch_functions)
            self.model = DDP(self.model)
        else:
            self.model = model
            self.optimizer = optimizer

        self._reset()

    def _reset(self):
        self.halving = False
        if self.args.continue_from:
            checkpoint = torch.load('logs/%s/model_dict.pt' % self.args.continue_from, map_location='cpu')

            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.args.distributed:
                self.amp.load_state_dict(checkpoint['amp'])

            self.start_epoch=checkpoint['epoch']
            self.prev_val_loss = checkpoint['prev_val_loss']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']

            if self.print: print("Resume training from epoch: {}".format(self.start_epoch))
            
        else:
            checkpoint = torch.load('../../pretrain_networks/sdr-av-dprnn-pretrain.pt', map_location='cpu')
            pretrained_model=checkpoint['model']
            self.model.load_state_dict(pretrained_model)

            self.prev_val_loss = float("inf")
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
            tr_loss = self._run_one_epoch(data_loader = self.train_data)
            tr_loss = self._reduce_tensor(tr_loss)

            if self.print: print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Train Loss {2:.3f}'.format(
                        epoch, time.time() - start, tr_loss))

            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(data_loader = self.validation_data, state='val')
                val_loss = self._reduce_tensor(val_loss)

            if self.print: print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Valid Loss {2:.3f}'.format(
                          epoch, time.time() - start, val_loss))

            # test
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                test_loss = self._run_one_epoch(data_loader = self.test_data, state='test')
                test_loss = self._reduce_tensor(test_loss)

            if self.print: print('Test Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Test Loss {2:.3f}'.format(
                          epoch, time.time() - start, test_loss))


            # Check whether to adjust learning rate and early stop
            if reduced_val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0

            # Halfing the learning rate
            self.halving = True
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] *0.98
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = reduced_val_loss

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar('Train_loss', tr_loss, epoch)
                    self.writer.add_scalar('Validation_loss', val_loss, epoch)
                    self.writer.add_scalar('Test_loss', test_loss, epoch)

                # Save model
                if reduced_val_loss < self.best_val_loss:
                    self.best_val_loss = reduced_val_loss
                    checkpoint = {'model': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'epoch': epoch+1,
                                    'prev_val_loss': self.prev_val_loss,
                                    'best_val_loss': self.best_val_loss,
                                    'val_no_impv': self.val_no_impv}
                    if self.args.distributed:
                        checkpoint{'amp':self.amp.state_dict()}
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict.pt")
                    print("Fund new best model, dict saved")

    def _run_one_epoch(self, data_loader, state='train'):
        total_loss = 0
        self.accu_count = 0
        for i, (a_mix, a_tgt, v_tgt) in enumerate(data_loader):
            a_mix = a_mix.cuda().squeeze(0).float()
            a_tgt = a_tgt.cuda().squeeze(0).float()
            v_tgt = v_tgt.cuda().squeeze(0).float()

            a_tgt_est = self.model(a_mix, v_tgt)

            pos_snr = cal_SDR(a_tgt, a_tgt_est)
            loss = 0 - torch.mean(pos_snr)

            # print(loss.item())
  
            if state == 'train':
                self.accu_count += 1
                if self.args.accu_grad:
                    loss = loss/(self.args.effec_batch_size / self.args.batch_size)
                    if self.args.distributed:
                        with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                                       self.args.max_norm)
                    else:
                        loss.backward()
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    if self.args.distributed:
                        with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                                       self.args.max_norm)
                    else:
                        loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad() 
                    
            if state=='test':
                loss = 0 - torch.mean(pos_snr[::self.args.C])

            total_loss += loss.data
            
        return total_loss / (i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt
