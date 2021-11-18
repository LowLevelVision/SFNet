import os
import time
import argparse

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from dataset import DeblurDataset
from models import model
import losses

torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False

def main():
    parser = argparse.ArgumentParser(description='SFNet')
    parser.add_argument('-e', '--epochs', type=int, default=3000, help='number of total epochs to run')
    parser.add_argument('-r', '--resume', type=bool, default=False, help='resume training')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('-s', '--image_size', type=int, default=256, help='training image size')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='initial learning rate')
    args = parser.parse_args()
    train(torch.device('cuda:0'), args)

def train(device, args):


    sfnet = model(num_resblocks=[7, 7, 7, 7], input_channels=[3, 6, 6, 6]).to(device)
    sfnet_optim = torch.optim.Adam(sfnet.parameters(), lr=args.learning_rate)

    warmup_epochs = 3
    scheduler_cosine = CosineAnnealingLR(sfnet_optim, args.epochs - warmup_epochs, eta_min=1e-6)
    sfnet_scheduler = GradualWarmupScheduler(sfnet_optim, multiplier=1, total_epoch=warmup_epochs,
                                                after_scheduler=scheduler_cosine)

    sfnet_optim.zero_grad()
    sfnet_optim.step()

    START_EPOCH = -1
    if args.resume:
        checkpoint = torch.load(str('./ckpts/checkpoint.pth'))
        sfnet_optim.load_state_dict(checkpoint['sfnet_optim'])
        START_EPOCH = checkpoint['epoch']
        for i in range(0, START_EPOCH + 1):
            sfnet_scheduler.step()
        print('load checkpoints success')
        sfnet.load_state_dict(torch.load(str('./ckpts/SFNet.pth')))
        print('load sfnet success')

    loss_fn = losses.CharbonnierLoss()

    if os.path.exists('./ckpts') == False:
        os.mkdir('./ckpts')

    train_dataset = DeblurDataset(
        root_dir='./dataset',
        blur_image_files='./dataset/train_blur_file.txt',
        sharp_image_files='./dataset/train_sharp_file.txt',
        rotation=True,
        adjust=True,
        crop=True,
        crop_size=args.image_size,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(START_EPOCH + 1, args.epochs):
        start = time.time()

        for target_image, input_image in train_dataloader:
            target_image = Variable(target_image - 0.5).to(device, non_blocking=True)
            input_image = Variable(input_image - 0.5).to(device, non_blocking=True)

            output_image = sfnet(input_image)[0]

            sfnet.zero_grad()
            loss = loss_fn(output_image + input_image, target_image)
            loss.backward()
            sfnet_optim.step()

        stop = time.time()
        print('epoch:', epoch, 'loss:%.6f' % loss.item(), 'time:%.4f' % (stop - start))

        sfnet_scheduler.step()

        state = {'sfnet_optim': sfnet_optim.state_dict(), 'epoch': epoch}
        torch.save(state, str('./ckpts/checkpoint.pth'))
        torch.save(sfnet.state_dict(), str('./ckpts/SFNet.pth'))

        if (epoch + 1) % 500 == 0:
            if os.path.exists('./ckpts/epoch' + str(epoch)) == False:
                os.mkdir('./ckpts/epoch' + str(epoch))
            torch.save(state, str('./ckpts/epoch' + str(epoch) + '/checkpoint.pth'))
            torch.save(sfnet.state_dict(), str('./ckpts/epoch' + str(epoch) + '/sfnet.pth'))

if __name__ == '__main__':
    main()