from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, GLUONTSDataset
from data_provider.uea import collate_fn
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    # 'm4': Dataset_M4,  Removed due to the LICENSE file constraints of m4.py
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    # datasets from gluonts package:
    "gluonts": GLUONTSDataset,
}


def random_subset(dataset, pct, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, idx[:int(len(dataset) * pct)].long().numpy())


def data_provider(args, config, flag, ddp=False):  # args,
    #Data：Custom/Glounts，从data_dict中获取定义好的Dataset类
    Data = data_dict[config['data']]
    timeenc = 0 if config['embed'] != 'timeF' else 1

    # 设置要传入Dataset类中的参数
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if 'anomaly_detection' in config['task_name']:  # 当前任务目标
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq


    #开始将参数逐个传入Dataset类中
    if 'gluonts' in config['data']:
        # process gluonts dataset:
        data_set = Data(
            dataset_name=config['dataset_name'],
            size=(config['seq_len'], config['label_len'], config['pred_len']),
            path=config['root_path'],
            # Don't set dataset_writer
            features=config["features"],
            flag=flag,
        )
        # 如果设置了子数据集采样百分比，并且当前是训练阶段，则进行子数据集采样
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader

    timeenc = 0 if config['embed'] != 'timeF' else 1

    if 'anomaly_detection' in config['task_name']:
        drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            win_size=config['seq_len'],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print("ddp mode is set to false for anomaly_detection", ddp, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)
        return data_set, data_loader
    elif 'classification' in config['task_name']:
        drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            flag=flag,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            sampler=DistributedSampler(data_set) if ddp else None,
            collate_fn=lambda x: collate_fn(x, max_len=config['seq_len'])
        )
        return data_set, data_loader
    else:
        if config['data'] == 'm4':
            drop_last = False
        data_set = Data(
            root_path=config['root_path'],
            data_path=config['data_path'],
            flag=flag,
            size=[config['seq_len'], config['label_len'], config['pred_len']],
            features=config['features'],
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=config['seasonal_patterns'] if config['data'] == 'm4' else None
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)
        return data_set, data_loader


class BalancedDataLoaderIterator:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

        self.num_dataloaders = len(dataloaders)

        #获取每个dataloader中样本数量最多的一个
        max_length = max(len(dataloader) for dataloader in dataloaders)     
        #获取每个dataloader的样本数量
        length_list = [len(dataloader) for dataloader in dataloaders]   #每个dataloader的规模 
        print("data loader length:", length_list)
        print("max dataloader length:", max_length,
              "epoch iteration:", max_length * self.num_dataloaders)    #打印每个epoch最多要迭代多少次
        #因为最大的数据集一共1146个batch，为了均衡，小数据集会被重置重新采样。
        self.total_length = max_length * self.num_dataloaders           #一个epoch中的总迭代次数<num_dataloader*max_len_dataloader
        self.current_iteration = 0
        self.probabilities = torch.ones(                                #采样概率，初始每个数据集均匀采样
            self.num_dataloaders, dtype=torch.float) / self.num_dataloaders

    def __iter__(self):
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self.current_iteration = 0
        return self

    def __next__(self):             #获取数据的核心方法
        if self.current_iteration >= self.total_length:     #当迭代次数=total,抛出异常
            raise StopIteration
        
        #根据概率随机选择一个数据加载器并采样，如果这个数据集的迭代器已经用尽，重新初始化。保证了即使某些数据集小，也能持续从所有数据集中采样。
        chosen_index = torch.multinomial(self.probabilities, 1).item()
        try:
            sample = next(self.iterators[chosen_index])
        except StopIteration:
            self.iterators[chosen_index] = iter(self.dataloaders[chosen_index])
            sample = next(self.iterators[chosen_index])

        self.current_iteration += 1
        return sample, chosen_index     #返回样本和对应的数据集标识
 
    def __len__(self):
        return self.total_length

    def generate_fake_samples_for_batch(self, dataloader_id, batch_size):#生成全0样本，用于确定每个任务能够支持的最大batch大小，在memory_check中使用
        if dataloader_id >= len(self.dataloaders) or dataloader_id < 0:
            raise ValueError("Invalid dataloader ID")

        dataloader = self.dataloaders[dataloader_id]
        iterator = iter(dataloader)

        try:
            sample_batch = next(iterator)
            fake_samples = []

            for sample in sample_batch:
                if isinstance(sample, torch.Tensor):
                    fake_sample = torch.zeros(
                        [batch_size] + list(sample.shape)[1:])
                    fake_samples.append(fake_sample)
                else:
                    pass

            return fake_samples, dataloader_id
        except StopIteration:
            return None
