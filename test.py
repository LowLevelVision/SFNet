import os
import time
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from models import model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False

CHECKPOINT = 'SFNet.pth'
TEST_SAMPLES = 'test_samples'
TEST_RESULTS = 'test_resutls'

def save_images(images, name):
    filename = TEST_RESULTS + '/' + name
    torchvision.utils.save_image(images, filename)

def main():
    sfnet = model(num_resblocks=[7, 7, 7, 7], input_channels=[3, 6, 6, 6]).to(device)
    sfnet.load_state_dict(torch.load(CHECKPOINT))
    print('load deblurnet success')

    if os.path.exists(TEST_RESULTS) == False:
        os.mkdir(TEST_RESULTS)

    # Hardware warm-up
    sfnet(torch.randn(1, 3, 256, 256).cuda())

    test_time = 0.0
    iteration = 1.0
    for images_name in os.listdir(TEST_SAMPLES):
        with torch.no_grad():
            input_image = transforms.ToTensor()(Image.open(TEST_SAMPLES + '/' + images_name).convert('RGB'))
            input_image = Variable(input_image-0.5).unsqueeze(0).to(device)

            start = time.time()
            d4 = sfnet(input_image)[0]
            stop = time.time()
            test_time += stop - start
            save_images(d4+input_image+0.5, images_name)
            print('RunTime:%.4f' % (stop - start), 'Average Runtime:%.4f' % (test_time / iteration))
            iteration += 1

if __name__ == '__main__':
    main()
