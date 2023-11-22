import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

import utils
from model import model as create_model
from utils import read_split_data, train_one_epoch, evaluate

#  tensorboard.exe --logdir=
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, input, out, transform=None):
        self.input = input
        self.out = out
        self.transform = transform

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        Input = self.input[item]
        Out = self.out[item]
        return Input, Out


def main(args):
    path = "./weights1"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists(path) is False:
        os.makedirs(path)
    data_train_input, data_valid_input, data_train_out, data_valid_out = read_split_data(args.path_in)
    train_dataset = MyDataSet(input=data_train_input, out=data_train_out)
    valid_dataset = MyDataSet(input=data_valid_input, out=data_valid_out)
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    train_data = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw)
    vali_data = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=nw)

    model = create_model(data_size=args.data_size, in_c=args.in_c, out_c=args.out_c,
                         conv_kernel_size=args.conv_kernel_size, conv_padding=args.conv_padding,
                         conv_stride=args.conv_stride,
                         trans_conv_kernel_size=args.trans_conv_kernel_size,
                         trans_conv_stride=args.trans_conv_stride, trans_conv_padding=args.trans_conv_padding,
                         embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5E-5, amsgrad=False)
    # 学习率衰减策略，lr=lr*x x=((1+cos(epoch*pi/epochs))/2)*(1-lrf)+lrf;
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    import time
    for epoch in range(args.epochs):
        # train
        start_time = time.time()
        train_one_epoch(model=model,
                        optimizer=optimizer,
                        data=train_data,
                        device=device,
                        epoch=epoch)
        end_time = time.time()
        elapsed_time = end_time - start_time
        scheduler.step()

        # validate
        evaluate(model=model,
                 data=vali_data,
                 device=device,
                 epoch=epoch)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            torch.save(model.state_dict(), path + "/model-{}.pt".format(epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)  # 0.001
    parser.add_argument('--lrf', type=float, default=0.01)
    in_c = 3
    out_c = 2
    data_size = (40, 120)
    conv_kernel_size = (5, 10)
    conv_stride = (1, 1)
    # conv_padding = tuple((conv_kernel_size[i] - 1) // 2 + 1 for i in range(len(data_size)))
    conv_padding = (3, 5)
    trans_conv_kernel_size = (5, 10)
    trans_conv_stride = (1, 1)
    # trans_conv_padding = tuple((trans_conv_kernel_size[i] - 1) // 2 + 1 for i in range(len(data_size)))
    trans_conv_padding = (3, 5)
    embed_dim = 20
    depth = 2
    num_heads = 4
    path_in = r'../data'
    parser.add_argument('--path-in', type=str,
                        default=path_in)
    parser.add_argument('--data-size', type=tuple,
                        default=data_size)
    parser.add_argument('--conv_kernel_size', type=tuple,
                        default=conv_kernel_size)
    parser.add_argument('--conv_stride', type=tuple,
                        default=conv_stride)
    parser.add_argument('--conv_padding', type=tuple,
                        default=conv_padding)
    parser.add_argument('--trans_conv_kernel_size', type=tuple,
                        default=trans_conv_kernel_size)
    parser.add_argument('--trans_conv_stride', type=tuple,
                        default=trans_conv_stride)
    parser.add_argument('--trans_conv_padding', type=tuple,
                        default=trans_conv_padding)
    parser.add_argument('--in_c', type=int,
                        default=in_c)
    parser.add_argument('--out_c', type=int,
                        default=out_c)
    parser.add_argument('--embed_dim', type=int,
                        default=embed_dim)
    parser.add_argument('--num_heads', type=int,
                        default=num_heads)
    parser.add_argument('--depth', type=int,
                        default=depth)
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default=r"",
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
