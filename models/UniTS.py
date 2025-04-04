"""
UniTS
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

from timm.layers import Mlp, DropPath
from timm.layers.helpers import to_2tuple

from .blocks import *

#输入 输入的序列长度、窗口，输出滑动窗口操作后的序列长度
def calculate_unfold_output_length(input_length, size, step):
    # Calculate the number of windows
    num_windows = (input_length - size) // step + 1
    return num_windows


class Model(nn.Module):
    """
    UniTS: Building a Unified Time Series Model
    """

    def __init__(self, args, configs_list, pretrain=False):
        super().__init__()

        #预训练掩码策略
        if pretrain:
            self.right_prob = args.right_prob
            self.min_mask_ratio = args.min_mask_ratio
            self.max_mask_ratio = args.max_mask_ratio

        # Tokens settings
        # 用字典存储不同类型的token，使用ParameterDict使其可训练
        self.num_task = len(configs_list)
        self.prompt_tokens = nn.ParameterDict({})
        self.mask_tokens = nn.ParameterDict({})
        self.cls_tokens = nn.ParameterDict({})
        self.category_tokens = nn.ParameterDict({})

        for i in range(self.num_task):
            #configs_list为[[task_name, task_config]...]
            dataset_name = configs_list[i][1]['dataset']
            task_data_name = configs_list[i][0]

            #按照数据集名称生成prompt_tokens
            if dataset_name not in self.prompt_tokens:
                #prompt_token的shape为[1, enc_in, prompt_num, d_model]， enc_in实际上是变量数
                self.prompt_tokens[dataset_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], args.prompt_num, args.d_model)
                torch.nn.init.normal_(
                    self.prompt_tokens[dataset_name], std=.02)
                self.mask_tokens[dataset_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)

            if configs_list[i][1]['task_name'] == 'classification':
                self.category_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], configs_list[i][1]['num_class'], args.d_model)
                torch.nn.init.normal_(
                    self.category_tokens[task_data_name], std=.02)
                self.cls_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)
                torch.nn.init.normal_(self.cls_tokens[task_data_name], std=.02)
            if pretrain:
                self.cls_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)
                torch.nn.init.normal_(self.cls_tokens[task_data_name], std=.02)

        self.cls_nums = {}
        for i in range(self.num_task):
            task_data_name = configs_list[i][0]
            if configs_list[i][1]['task_name'] == 'classification':
                self.cls_nums[task_data_name] = configs_list[i][1]['num_class']
            elif configs_list[i][1]['task_name'] == 'long_term_forecast':
                remainder = configs_list[i][1]['seq_len'] % args.patch_len
                if remainder == 0:
                    padding = 0
                else:
                    padding = args.patch_len - remainder
                input_token_len = calculate_unfold_output_length(
                    configs_list[i][1]['seq_len']+padding, args.stride, args.patch_len)
                input_pad = args.stride * \
                    (input_token_len - 1) + args.patch_len - \
                    configs_list[i][1]['seq_len']
                pred_token_len = calculate_unfold_output_length(
                    configs_list[i][1]['pred_len']-input_pad, args.stride, args.patch_len)
                real_len = configs_list[i][1]['seq_len'] + \
                    configs_list[i][1]['pred_len']
                self.cls_nums[task_data_name] = [pred_token_len,
                                                 configs_list[i][1]['pred_len'], real_len]

        self.configs_list = configs_list

        ### model settings ###
        self.prompt_num = args.prompt_num
        self.stride = args.stride
        self.pad = args.stride
        self.patch_len = args.patch_len

        # input processing
        self.patch_embeddings = PatchEmbedding(
            args.d_model, args.patch_len, args.stride, args.stride, args.dropout)
        self.position_embedding = LearnablePositionalEmbedding(args.d_model)
        self.prompt2forecat = DynamicLinear(128, 128, fixed_in=args.prompt_num)

        # basic blocks
        self.block_num = args.e_layers
        self.blocks = nn.ModuleList(
            [BasicBlock(dim=args.d_model, num_heads=args.n_heads, qkv_bias=False, qk_norm=False,
                        mlp_ratio=8., proj_drop=args.dropout, attn_drop=0., drop_path=0.,
                        init_values=None, prefix_token_length=args.prompt_num) for l in range(args.e_layers)]
        )

        # output processing
        self.cls_head = CLSHead(args.d_model, head_dropout=args.dropout)
        self.forecast_head = ForecastHead(
            args.d_model, args.patch_len, args.stride, args.stride, prefix_token_length=args.prompt_num, head_dropout=args.dropout)
        if pretrain:
            self.pretrain_head = ForecastHead(
                args.d_model, args.patch_len, args.stride, args.stride, prefix_token_length=1, head_dropout=args.dropout)

    def tokenize(self, x, mask=None):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x * x, dim=1) /
                               torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(dim=1)
        else:
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        x = x.permute(0, 2, 1)
        remainder = x.shape[2] % self.patch_len
        if remainder != 0:
            padding = self.patch_len - remainder
            x = F.pad(x, (0, padding))
        else:
            padding = 0
        x, n_vars = self.patch_embeddings(x)
        return x, means, stdev, n_vars, padding

    def prepare_prompt(self, x, n_vars, prefix_prompt, task_prompt, task_prompt_num, task_name=None, mask=None):
        x = torch.reshape(
            x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        # append prompt tokens
        this_prompt = prefix_prompt.repeat(x.shape[0], 1, 1, 1)

        if task_name == 'forecast':
            this_mask_prompt = task_prompt.repeat(
                x.shape[0], 1, task_prompt_num, 1)
            init_full_input = torch.cat(
                (this_prompt, x, this_mask_prompt), dim=-2)
            init_mask_prompt = self.prompt2forecat(init_full_input.transpose(
                -1, -2), init_full_input.shape[2]-prefix_prompt.shape[2]).transpose(-1, -2)
            this_function_prompt = init_mask_prompt[:, :, -task_prompt_num:]
            x = torch.cat((this_prompt, x, this_function_prompt), dim=2)
            x[:, :, self.prompt_num:] = x[:, :, self.prompt_num:] + \
                self.position_embedding(x[:, :, self.prompt_num:])
        elif task_name == 'classification':
            this_function_prompt = task_prompt.repeat(x.shape[0], 1, 1, 1)
            x = x + self.position_embedding(x)
            x = torch.cat((this_prompt, x, this_function_prompt), dim=2)
        elif task_name == 'imputation':
            # fill the masked parts with mask tokens
            # for imputation, masked is 0, unmasked is 1, so here to reverse mask
            mask = 1-mask
            mask = mask.permute(0, 2, 1)
            mask = self.mark2token(mask)
            mask_repeat = mask.unsqueeze(dim=-1)

            mask_token = task_prompt
            mask_repeat = mask_repeat.repeat(1, 1, 1, x.shape[-1])
            x = x * (1-mask_repeat) + mask_token * mask_repeat

            init_full_input = torch.cat((this_prompt, x), dim=-2)
            init_mask_prompt = self.prompt2forecat(
                init_full_input.transpose(-1, -2), x.shape[2]).transpose(-1, -2)
            # keep the unmasked tokens and fill the masked ones with init_mask_prompt.
            x = x * (1-mask_repeat) + init_mask_prompt * mask_repeat
            x = x + self.position_embedding(x)
            x = torch.cat((this_prompt, x), dim=2)
        elif task_name == 'anomaly_detection':
            x = x + self.position_embedding(x)
            x = torch.cat((this_prompt, x), dim=2)

        return x

    def mark2token(self, x_mark):
        x_mark = x_mark.unfold(
            dimension=-1, size=self.patch_len, step=self.stride)
        x_mark = x_mark.mean(dim=-1)
        x_mark = (x_mark > 0).float()
        return x_mark

    def backbone(self, x, prefix_len, seq_len):
        attn_mask = None
        for block in self.blocks:
            x = block(x, prefix_seq_len=prefix_len +
                      seq_len, attn_mask=attn_mask)
        return x

    def forecast(self, x, x_mark, task_id):
        dataset_name = self.configs_list[task_id][1]['dataset']
        task_data_name = self.configs_list[task_id][0]
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.mask_tokens[dataset_name]
        task_prompt_num = self.cls_nums[task_data_name][0]
        task_seq_num = self.cls_nums[task_data_name][1]
        real_seq_len = self.cls_nums[task_data_name][2]

        x, means, stdev, n_vars, _ = self.tokenize(x)

        x = self.prepare_prompt(
            x, n_vars, prefix_prompt, task_prompt, task_prompt_num, task_name='forecast')

        seq_token_len = x.shape[-2]-prefix_prompt.shape[2]
        
        #在这里经过backbone模块
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        x = self.forecast_head(
            x, real_seq_len, seq_token_len)
        x = x[:, -task_seq_num:]

        # De-Normalization from Non-stationary Transformer
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        return x

    def classification(self, x, x_mark, task_id):
        dataset_name = self.configs_list[task_id][1]['dataset']
        task_data_name = self.configs_list[task_id][0]
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.cls_tokens[task_data_name]
        task_prompt_num = 1
        category_token = self.category_tokens[task_data_name]

        x, means, stdev, n_vars, _ = self.tokenize(x)

        seq_len = x.shape[-2]

        x = self.prepare_prompt(
            x, n_vars, prefix_prompt, task_prompt, task_prompt_num, task_name='classification')

        x = self.backbone(x, prefix_prompt.shape[2], seq_len)

        x = self.cls_head(x, category_token)

        return x

    def imputation(self, x, x_mark, mask, task_id):
        dataset_name = self.configs_list[task_id][1]['dataset']
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.mask_tokens[dataset_name]

        seq_len = x.shape[1]
        x, means, stdev, n_vars, padding = self.tokenize(x, mask)

        x = self.prepare_prompt(
            x, n_vars, prefix_prompt, task_prompt, None, mask=mask, task_name='imputation')
        seq_token_len = x.shape[-2]-prefix_prompt.shape[2]
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        x = self.forecast_head(
            x, seq_len+padding, seq_token_len)
        x = x[:, :seq_len]

        # De-Normalization from Non-stationary Transformer
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        return x

    def anomaly_detection(self, x, x_mark, task_id):
        dataset_name = self.configs_list[task_id][1]['dataset']
        prefix_prompt = self.prompt_tokens[dataset_name]

        seq_len = x.shape[1]
        x, means, stdev, n_vars, padding = self.tokenize(x)

        x = self.prepare_prompt(x, n_vars, prefix_prompt,
                                None, None, task_name='anomaly_detection')
        seq_token_len = x.shape[-2]-prefix_prompt.shape[2]
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        x = self.forecast_head(
            x, seq_len+padding, seq_token_len)
        x = x[:, :seq_len]

        # De-Normalization from Non-stationary Transformer
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        return x



    def pretraining(self, x, x_mark, task_id, enable_mask=False):
        dataset_name = self.configs_list[task_id][1]['dataset']
        task_data_name = self.configs_list[task_id][0]

        prefix_prompt = self.prompt_tokens[dataset_name]
        mask_token = self.mask_tokens[dataset_name]
        cls_token = self.cls_tokens[task_data_name]

        seq_len = x.shape[1]
        #x[batch_size * n_vars, seq_token_len, d_model],
        """means：形状为 [batch_size, 1, feature_dim]，序列均值
            stdev：形状为 [batch_size, 1, feature_dim]，序列标准差
            n_vars：标量，变量数量（时间序列中的特征数）
            padding：标量，为使序列长度可被patch_len整除而添加的填充长度"""
        x, means, stdev, n_vars, padding = self.tokenize(x)
        seq_token_len = x.shape[-2]

        # 输入序列变为[batch_size, n_vars变量数, seq_token_len分块数目, d_model]
        x = torch.reshape(
            x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        #  [batch_size, n_vars, prompt_num, d_model]
        this_prompt = prefix_prompt.repeat(x.shape[0], 1, 1, 1)

        #如果启用掩码，将被掩码的位置替换为掩码token，添加位置编码，加上提示token
        if enable_mask:
            mask = self.choose_masking(x, self.right_prob,
                                       self.min_mask_ratio, self.max_mask_ratio)
            #mask[batch_size, seq_token_len]
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x.shape[1], 1, x.shape[-1])
            #mask_repeat[batch_size, n_vars, seq_token_len, d_model]扩张维度，
            #需要掩码的位置为1，不需要的位置是0

            
            x = x * (1-mask_repeat) + mask_token * mask_repeat
            #在不需要掩码的位置为原始值，掩码位置为mask_token

            init_full_input = torch.cat((this_prompt, x), dim=-2)
            init_mask_prompt = self.prompt2forecat(
                init_full_input.transpose(-1, -2), x.shape[2]).transpose(-1, -2)
            # 用预测值代替掩码token，并添加位置编码
            x = x * (1-mask_repeat) + init_mask_prompt * mask_repeat
            x = x + self.position_embedding(x)

            # 将token形式的掩码转换回序列位置上，用于之后的损失计算
            mask_seq = self.get_mask_seq(mask, seq_len+padding)
            mask_seq = mask_seq[:, :seq_len]
        this_function_prompt = cls_token.repeat(x.shape[0], 1, 1, 1)

        """prefix_prompt [1, enc_in, prompt_num, d_model]
            this_prompt[batch_size,enc_in,prompt_num,d_model]
            this_function_prompt:[batch_size, enc_in, 1, d_model]"""
        x = torch.cat((this_prompt, x, this_function_prompt), dim=2)
        #最后连接一个CLStoken
        #连接之后经过backbone模块
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        #双路径重建输出
        if enable_mask:
            # x = [batch_size, 变量数, token序列长度, d_model]
            #forecast_head == GEN tower
            # 输出 [batch_size, seq_len+padding, 变量数]（已转换回时间序列格式）
            mask_dec_out = self.forecast_head(
                x[:, :, :-1], seq_len+padding, seq_token_len)
            mask_dec_out = mask_dec_out[:, :seq_len]
            
            # 对预测输出反标准化，还原到原始数据尺度
            mask_dec_out = mask_dec_out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(
                    1, mask_dec_out.shape[1], 1))
            mask_dec_out = mask_dec_out + \
                (means[:, 0, :].unsqueeze(1).repeat(
                    1, mask_dec_out.shape[1], 1))
            
            #使用CLS_HEAD提取分类特征。[batch_size, 变量数, 1, d_model]。
            cls_dec_out = self.cls_head(x, return_feature=True)
            # detach样本token，连接cls_token，只更新cls_token
            fused_dec_out = torch.cat(
                (cls_dec_out, x[:, :, self.prompt_num:-1].detach()), dim=2)
            
            #使用预测头进行另一种预测[batch_size,seq_len+padding,num_vars]
            cls_dec_out = self.pretrain_head(
                fused_dec_out, seq_len+padding, seq_token_len)
            cls_dec_out = cls_dec_out[:, :seq_len]
            cls_dec_out = cls_dec_out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(
                    1, cls_dec_out.shape[1], 1))
            cls_dec_out = cls_dec_out + \
                (means[:, 0, :].unsqueeze(1).repeat(
                    1, cls_dec_out.shape[1], 1))

            #对cls_token预测结果进行反标准化，输出形状仍为[batch_size, seq_len, 变量数]
            return cls_dec_out, mask_dec_out, mask_seq
        else:
            return cls_dec_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                mask=None, task_id=None, task_name=None, enable_mask=None):
        if task_name == 'long_term_forecast' or task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, task_id)
            return dec_out  # [B, L, D]
        if task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, mask, task_id)
            return dec_out  # [B, L, D]
        if task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc, task_id)
            return dec_out  # [B, L, D]
        if task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc, task_id)
            return dec_out  # [B, N]
        if 'pretrain' in task_name:
            dec_out = self.pretraining(x_enc, x_mark_enc, task_id,
                                       enable_mask=enable_mask)
            return dec_out
        return None
