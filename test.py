import os
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, models
from model import *
import pretrainedmodels


DATA_ROOT = './testimg'
RESULT_FILE = 'result.csv'

def test_and_generate_result_round2(epoch_num='1', model_name='resnet101', img_size=320):
    data_transform = transforms.Compose([
        transforms.Resize(img_size, Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize([0.53744068, 0.51462684, 0.52646497], [0.06178288, 0.05989952, 0.0618901])
    ])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    is_use_cuda = torch.cuda.is_available()


    model_hmy = torch.load('./checkpoint/' + model_name + '/Models_epoch_' + epoch_num + '.ckpt',
                            map_location=lambda storage, loc: storage.cuda())
    if is_use_cuda:
        model_hmy =model_hmy.cuda()
    model_hmy.eval()


    with open(os.path.join('checkpoint', model_name, model_name+'_'+str(img_size)+'_'+RESULT_FILE), 'w', encoding='utf-8') as fd:
        fd.write('filename|defect,probability\n')
        test_files_list = os.listdir(DATA_ROOT)
        for _file in test_files_list:
            file_name = _file
            if '.jpg' not in file_name:
                continue
            file_path = os.path.join(DATA_ROOT, file_name)
            img_tensor = data_transform(Image.open(file_path).convert('RGB')).unsqueeze(0)
            if is_use_cuda:
                img_tensor = Variable(img_tensor.cuda(), volatile=True)
            _, output, _ = model_hmy(img_tensor)
            #output = my_model(img_tensor)
            output = F.softmax(output, dim=1)
            for k in range(11):
                defect_prob = round(output.data[0, k], 6)
                if defect_prob == 0.:
                    defect_prob = 0.000001
                elif defect_prob == 1.:
                    defect_prob = 0.999999
                target_str = '%s,%.6f\n' % (file_name + '|' + ('norm' if 0 == k else 'defect_'+str(k)), defect_prob)
                fd.write(target_str)

if __name__ == '__main__':

    test_and_generate_result_round2()
