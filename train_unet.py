import logging
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from unet import UNet
from torch.utils.data import DataLoader, random_split
from dataloaders.unet_loader import UnetLoader
from apex import amp
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    TrainNum = 5000
    TestNum = 6226 - TrainNum
    BatchSize = 8
    Epoch = 1500
    Lr = 0.001
    ImageSize = 512
    Number_of_runs = 1

    writer = SummaryWriter(comment=f'lr_{Lr}_BS_{BatchSize}_ImageSize_{ImageSize}_Number_of_runs{Number_of_runs}')
    global_step = 0
    savefile_dirpath = r'D\results'
    savefile_dirpath = savefile_dirpath + str(Number_of_runs)
    if not os.path.exists(savefile_dirpath):
        os.makedirs(savefile_dirpath)


    dataset = UnetLoader(r'C:\Users\Tim Wang\Desktop\data\train_data',
                         r'C:\Users\Tim Wang\Desktop\data\train_label',
                         r'C:\Users\Tim Wang\Desktop\data\cache',
                         ImgSize=ImageSize)
    train, val = random_split(dataset, [TrainNum, TestNum])
    TrainLoader = DataLoader(train, batch_size=BatchSize, shuffle=True, num_workers=8, pin_memory=False)
    ValLoader = DataLoader(val, batch_size=BatchSize, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)

    logging.info(f'''Starting training:
        Epochs:          {Epoch}
        Batch size:      {BatchSize}
        Learning rate:   {Lr}
        Training size:   {TrainNum}
        Validation size: {TestNum}
    ''')

    model = UNet(n_channels=4, n_classes=1, bilinear=True)
    model = model.to('cuda:0')
    optimizer = optim.SGD(model.parameters(), lr=Lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 500, 750, 1000, 1250], gamma=0.2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)
    criterion = nn.BCEWithLogitsLoss().cuda()

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model = nn.DataParallel(model, device_ids=[0])
    model.train()

    for epoch in range(Epoch):
        iter = 0
        epoch_loss = 0
        for batch in tqdm(TrainLoader):
            images = batch['image']
            label = batch['label']
            fpred = batch['fpred']
            MultiChannelInput = torch.cat((images, fpred), dim=1)
            label = label.to(device='cuda:0', dtype=torch.float32)
            MultiChannelInput = MultiChannelInput.to(device='cuda:0', dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(MultiChannelInput)

            # loss = criterion(pred, label.squeeze(dim=1))
            loss = criterion(pred, label)

            loss = loss * 5

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)

            epoch_loss += loss.item()

            iter += 1
            global_step += 1
            epoch_loss += loss.item()
            if (iter + 1) % 200 == 0:
                print('In iter:', iter + 1, 'current avg loss is:', epoch_loss / (iter + 1))

                fig = plt.figure()
                ax1 = fig.add_subplot(2, 2, 1)
                ax1.imshow(label[0][0].detach().cpu(), cmap='gray')
                ax2 = fig.add_subplot(2, 2, 2)
                ax2.imshow(label[1][0].detach().cpu(), cmap='gray')
                ax3 = fig.add_subplot(2, 2, 3)
                ax3.imshow(pred[0][0].detach().cpu(), cmap='gray')
                ax4 = fig.add_subplot(2, 2, 4)
                ax4.imshow(pred[1][0].detach().cpu(), cmap='gray')
                plt.show()
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('average loss', epoch_loss / (iter + 1), global_step)
                for minibatch in pred:
                    writer.add_image('predicted image', minibatch[0].unsqueeze(dim=0), global_step)
                writer.close()


        with torch.no_grad():
            loss_total = 0
            for i, batch in tqdm(enumerate(ValLoader)):
                images = batch['image']
                label = batch['label']
                fpred = batch['fpred']
                MultiChannelInput = torch.cat((images, fpred), dim=1)
                label = label.to(device='cuda:0', dtype=torch.float32)
                MultiChannelInput = MultiChannelInput.to(device='cuda:0', dtype=torch.float32)

                pred = model(MultiChannelInput)
                # loss = criterion(pred, label.squeeze(dim=1))
                loss = criterion(pred, label)

                loss = loss * 5

                loss_total += loss.item()
                if i == len(ValLoader) - 1:
                    for minibatch in pred:
                        writer.add_image('predicted image', minibatch[0].unsqueeze(dim=0), global_step)
                    fig = plt.figure()
                    ax1 = fig.add_subplot(2, 2, 1)
                    ax1.imshow(label[0][0].detach().cpu(), cmap='gray')
                    ax2 = fig.add_subplot(2, 2, 2)
                    ax2.imshow(label[1][0].detach().cpu(), cmap='gray')
                    ax3 = fig.add_subplot(2, 2, 3)
                    ax3.imshow(pred[0][0].detach().cpu(), cmap='gray')
                    ax4 = fig.add_subplot(2, 2, 4)
                    ax4.imshow(pred[1][0].detach().cpu(), cmap='gray')
                    plt.show()

            print(f'The length of valload is: {TestNum}', f'and the average loss is: {loss_total / 153}')
            writer.add_scalar('val_loss', loss_total / TestNum, epoch)
            writer.close()
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(savefile_dirpath, f'{epoch + 1}.pth'))
        scheduler.step(epoch)










