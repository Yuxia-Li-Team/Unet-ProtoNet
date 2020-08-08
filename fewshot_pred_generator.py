import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model.ProtoNet import FewShotSeg
from utils.utils import set_seed
from config import ex
import tqdm
from apex import amp
from dataloaders.to_harddrive_reader import reader
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
from torchvision.utils import save_image

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(0)
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(in_channels=3,
                       pretrained_path=_config['path']['init_path'],
                       cfg={'align': True})
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0,])
    model.load_state_dict(torch.load(r'D:\Pycharm Projects\PANet RSI\runs\PANet_VOC_align_sets_2_1way_5shot_[train]\503\snapshots\40000.pth',
                                     map_location='cpu'))
    model.eval()

    _log.info('###### Load data ######')
    make_data = reader
    dataset = make_data(dir=r'C:/Users/Tim Wang/Desktop/data/',
                        iter=6226,
                        shot_count=5,
                        query_count=1,
                        batch_size=1,
                        size=844)

    trainloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=False,
        drop_last=True
    )

    _log.info('###### Training ######')
    with torch.no_grad():
        for i_iter, sample_batched in tqdm.tqdm(enumerate(trainloader)):
            # Prepare input
            support_images = [[sample['image'].cuda()
                              for sample in sample_batched[:5]]]
            support_fg_mask = [[sample['fg'].cuda()
                               for sample in sample_batched[:5]]]
            support_bg_mask = [[sample['bg'].cuda()
                               for sample in sample_batched[:5]]]
            query_images = [sample_batched[-1]['image'].cuda()]
            query_labels = [sample_batched[-1]['fg'].cuda()]

            # Forward
            query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask,
                                           query_images)
            print('in iter:', i_iter, 'processing', sample_batched[-1]['name'])
            # figure = plt.figure()
            # ax1 = figure.add_subplot(2, 3, 1)
            # ax1.imshow(support_fg_mask[0][0].detach().cpu()[0], cmap='gray')
            # ax2 = figure.add_subplot(2, 3, 2)
            # ax2.imshow(support_fg_mask[0][1].detach().cpu()[0], cmap='gray')
            # ax3 = figure.add_subplot(2, 3, 3)
            # ax3.imshow(support_fg_mask[0][2].detach().cpu()[0], cmap='gray')
            # ax4 = figure.add_subplot(2, 3, 4)
            # ax4.imshow(support_fg_mask[0][3].detach().cpu()[0], cmap='gray')
            # ax5 = figure.add_subplot(2, 3, 5)
            # ax5.imshow(support_fg_mask[0][4].detach().cpu()[0], cmap='gray')
            # ax6 = figure.add_subplot(2, 3, 6)
            # ax6.imshow(query_pred.detach().argmax(dim=1)[0].cpu().numpy(), cmap='gray')
            # plt.show()
            save_image(query_pred.detach().cpu()[0][0].unsqueeze(dim=0),
                       r'C:\Users\Tim Wang\Desktop\data\cache\{}'.format(sample_batched[-1]['name'][0][:-8] + '_fewshotpred.png'),
                       normalize=True, scale_each=True)
            # query_pred = ToPILImage()(query_pred.detach().cpu()[0][1])
            # figure2 = plt.figure()
            # ax1 = figure2.add_subplot(1, 1, 1)
            # ax1.imshow(query_pred)
            # plt.show()
            # query_pred.save(r'C:\Users\Tim Wang\Desktop\data\cache\{}'.format(sample_batched[-1]['name'][0][:-8] + '_fewshotpred.jpg'))
