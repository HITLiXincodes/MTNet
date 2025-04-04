from __future__ import print_function
import torch
from PIL import Image
from torch.autograd import Variable
from utils.timer import Timer
from utils.logger import Logger
from utils import utils
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import os
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
if __name__ == '__main__':

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])   
        
    def load_img(filepath):
        image = Image.open(filepath).convert('RGB')
        return  image

    opt = TrainOptions().parse()

    dataset = create_dataset(opt)  
    dataset_size = len(dataset)   
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
        
    logger = Logger(opt)
    timer = Timer()
    to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    single_epoch_iters = (dataset_size // opt.batch_size)
    total_iters = opt.total_epochs * single_epoch_iters 
    cur_iters = opt.resume_iter + opt.resume_epoch * single_epoch_iters
    start_iter = opt.resume_iter
    print('Start training from epoch: {:05d}; iter: {:07d}'.format(opt.resume_epoch, opt.resume_iter))
for epoch in range(opt.resume_epoch, opt.total_epochs + 1):    
    for i, data in enumerate(dataset, start=start_iter):
        cur_iters += 1
        logger.set_current_iter(cur_iters)
        # =================== load data ===============
        model.set_input(data, cur_iters)
        timer.update_time('DataTime')

        # =================== model train ===============
        model.forward()
        timer.update_time('Forward')
        model.optimize_parameters()
        timer.update_time('Backward')
        loss = model.get_current_losses()
        loss.update(model.get_lr())
        logger.record_losses(loss)

        # =================== save model and visualize ===============
        if cur_iters % opt.print_freq == 0:
            print('Model log directory: {}'.format(opt.expr_dir))
            epoch_progress = '{:03d}|{:05d}/{:05d}'.format(epoch, i, single_epoch_iters)
            logger.printIterSummary(epoch_progress, cur_iters, total_iters, timer)

        if cur_iters % opt.visual_freq == 0:
            visual_imgs = model.get_current_visuals()
            logger.record_images(visual_imgs)

        info = {'resume_epoch': epoch, 'resume_iter': i+1}
        if cur_iters % opt.save_iter_freq == 0 and cur_iters>=160000:
            print('saving current model (epoch %d, iters %d)' % (epoch, cur_iters))
            if not os.path.exists("result/"+str(cur_iters)):
                os.makedirs("result/"+str(cur_iters))

            if not os.path.exists("result_coarse/"+str(cur_iters)):
                os.makedirs("result_coarse/"+str(cur_iters))
            save_suffix = 'iter_%d' % cur_iters 
            model.save_networks(save_suffix, info)
            avg_psnr = 0
            image_ldir = "/home/ubuntu/thumb_data/ce_test_en2_4/"
            image_filenames = [x for x in os.listdir(image_ldir) if is_image_file(x)]
            transform_list = [transforms.ToTensor()]
            transform = transforms.Compose(transform_list)
            for image_name in image_filenames:
                imgg = load_img(image_ldir + image_name)
                imgg = imgg.resize((32,32),Image.BICUBIC)
                img = imgg.resize((128, 128), Image.BICUBIC)

                input = to_tensor(img)
                with torch.no_grad():
                    input = Variable(input).view(1, -1, 128, 128)
                network = model.netG
                network.eval()
                out1 , out = network(input)

                output_sr_img = utils.tensor_to_img(out, normal=True)
                save_img = Image.fromarray(output_sr_img)
                save_img.save("result/{}/{}".format(cur_iters, image_name))

                output_sr_img = utils.tensor_to_img(out1, normal=True)
                save_img = Image.fromarray(output_sr_img)
                save_img.save("result_coarse/{}/{}".format(cur_iters, image_name))

        if cur_iters % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, iters %d)' % (epoch, cur_iters))
            model.save_networks('latest', info)

        if opt.debug: break
    if opt.debug and epoch > 5: exit()

    
    



