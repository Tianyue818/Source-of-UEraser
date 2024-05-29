import time
import models
import torch
import util
import kornia.augmentation as K
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Trainer():
    def __init__(self, criterion, data_loader, logger, config, global_step=0,
                 target='train_dataset'):
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.global_step = global_step
        self.target = target
        print(self.target)

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def train(self, epoch, model, criterion, optimizer, repeat, random_noise=None):
        model.train()
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if random_noise is not None:
                random_noise = random_noise.detach().to(device)
                for i in range(len(labels)):
                    class_index = labels[i].item()
                    images[i] += random_noise[class_index].clone()
                    images[i] = torch.clamp(images[i], 0, 1)
            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer, epoch, repeat)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step

    def train_batch(self, images, labels, model, optimizer, epoch, repeat):
        model.zero_grad()
        optimizer.zero_grad()
        UEraser = K.AugmentationSequential(
            K.RandomPlasmaBrightness(roughness=(0.3, 0.7), intensity=(0.5, 1.0), same_on_batch=False, p=1.0, keepdim=True),
            K.RandomPlasmaContrast(roughness=(0.3, 0.7), p=1.0),
            K.RandomChannelShuffle(same_on_batch=False, p=0.5, keepdim=True),
            K.auto.TrivialAugment()
        )
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            result_tensor = torch.empty((5, images.shape[0]))
            # UEraser
            if epoch < 50:
                for i in range(repeat):
                    images_tmp = UEraser(images)
                    logits_tmp = model(images_tmp)
                    loss_tmp = F.cross_entropy(logits_tmp, labels, reduction='none')
                    result_tensor[i] = loss_tmp
                logits = logits_tmp
                max_values, _ = torch.max(result_tensor, dim=0)
                loss = torch.mean(max_values)
            # UEraser-fast
            else:
                images = UEraser(images)
                logits = model(images)
                loss = self.criterion(logits, labels)
        else:
            logits, loss = self.criterion(model, images, labels, optimizer)
        if isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            _, labels = torch.max(labels.data, 1)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()
        if logits.shape[1] >= 5:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc, acc5 = acc.item(), acc5.item()
        else:
            acc, = util.accuracy(logits, labels, topk=(1,))
            acc, acc5 = acc.item(), 1
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])
        self.acc5_meters.update(acc5, labels.shape[0])
        payload = {"acc": acc,
                   "acc_avg": self.acc_meters.avg,
                   "loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload
