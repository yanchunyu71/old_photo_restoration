import numpy as np
import paddle
from Global.options.train_options import TrainOptions
import Global.models.networks as networks
from Global.models.models import create_model

# 导入reprod_log中的ReprodLogger类
from reprod_log import ReprodLogger

reprod_logger = ReprodLogger()
# 组网并初始化
# 只保存了参数
opt = TrainOptions().parse()

netD = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D,
            opt, opt.norm, opt.no_lsgan, opt.num_D, 
            not opt.no_ganFeat_loss, gpu_ids=opt.gpu_ids)
#print(3,64,3,opt,instance False 2 True [0])
pthfile = "work/oldphoto_paddle_D.pdparams"
netD.load_state_dict(paddle.load(pthfile))

netD.eval()

# 读入fake data并转换为tensor
fake_data = np.load("/content/drive/MyDrive/oldphoto/Global/fake_label.npy")

input1 = paddle.to_tensor(fake_data)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#input1 = input1.to(device)
# 模型前向
out = netD(input1)
np_out = np.array(out)
# 保存前向结果，对于不同的任务，需要开发者添加。
reprod_logger.add("logits", out.cpu().detach().numpy())
reprod_logger.save("forward_paddle.npy")
