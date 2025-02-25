from typing import Union, Tuple
import torch
import abc

# 控制注意力层的行为，利用 abc 模块来标识它是一个抽象基类，意味着它本身不能被实例化，而是需要被子类继承
class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def between_steps(self):
        return

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    # 抽象方法，用 @abc.abstractmethod 装饰，意味着任何 AttentionControl 的子类都必须实现这个方法
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0: 
            h = attn.shape[0]
            self.forward(attn[h // 2:], is_cross, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn


# 用于存储和解压注意力张量
class AttentionStore(AttentionControl):
    def __init__(self, res):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {} # 存储所有步骤的累积注意力值
        self.res = res


    # 返回一个空的存储字典，包含了各个注意力类型的空列表
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}


    # 根据传入的 attn 张量，is_cross，以及在 U-Net 中的位置（place_in_unet），将注意力值存储在 step_store 字典中相应的位置
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.res // 16) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        # 如果 attention_store 是空的，则直接将 step_store 赋值给 attention_store
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            # 在每个步骤之间，将 step_store 中的注意力值累积到 attention_store 中
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.step_store[key][i] + self.attention_store[key][i]
        self.step_store = self.get_empty_store()

    # 计算并返回每个注意力类型的平均注意力值
    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


# 控制和编辑注意力机制
class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(self, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]], res):
        super(AttentionControlEdit, self).__init__(res)
        self.batch_size = 2
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps  #自注意力替换的步骤范围
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.loss = 0
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion = torch.nn.KLDivLoss()


    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 调用了父类的 forward 方法以保存注意力值
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if not is_cross:
                self.loss += self.criterion(attn[1:], self.replace_self_attention(attn_base, attn_repalce))
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def replace_self_attention(self, attn_base, att_replace):
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
