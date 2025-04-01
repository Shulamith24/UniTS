import torch


class BalancedDataLoaderIterator:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

        self.num_dataloaders = len(dataloaders)

        max_length = max(len(dataloader) for dataloader in dataloaders)     

        length_list = [len(dataloader) for dataloader in dataloaders]   #每个dataloader的规模 
        print("data loader length:", length_list)
        print("max dataloader length:", max_length,
              "epoch iteration:", max_length * self.num_dataloaders)
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
