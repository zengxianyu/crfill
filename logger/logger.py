from jinja2 import Environment, FileSystemLoader
import os
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pdb

palette_ade = \
	[120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140, 140, 204, 5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82, 143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255, 255, 7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184, 6, 10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0, 255, 20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10, 15, 20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255, 31, 0, 255, 11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163, 0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255, 173, 255, 0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0, 31, 255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255, 0, 194, 255, 0, 122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0, 255, 163, 255, 0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184, 255, 0, 214, 255, 255, 0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153, 0, 255, 71, 255, 0, 255, 0, 163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0, 10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204, 41, 0, 255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0, 133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 92, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

class Logger:
    def __init__(self, log_dir=None, clear=False, palette=palette_ade, port=None):
        if log_dir is not None:
            # color palette
            #colors = loadmat('color150.mat')['colors']
            #palette = colors.reshape(-1)
            #palette = list(palette)
            #palette += ([0] * (256*3-len(palette)))
            #pdb.set_trace()
            self.palette = palette

            self.log_dir = log_dir
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.log_dir = log_dir
            self.plot_dir = os.path.join(log_dir, "plot")
            if not os.path.exists(self.plot_dir):
                os.mkdir(self.plot_dir)
            elif clear:
                os.system("rm {}/plot/*".format(log_dir))
            self.image_dir = os.path.join(log_dir, "image")
            if not os.path.exists(self.image_dir):
                os.mkdir(self.image_dir)
            elif clear:
                os.system("rm -rf {}/image/*".format(log_dir))
            if not os.path.exists(os.path.join(log_dir, "image_ticks")):
                os.mkdir(os.path.join(log_dir, "image_ticks"))
            elif clear:
                os.system("rm -rf {}/image_ticks/*".format(log_dir))
            self.plot_vals = {}
            self.plot_times = {}
            #def http_server():
            #    Handler = QuietHandler
            #    with socketserver.TCPServer(("", port), Handler) as httpd:
            #        #print("serving at port", PORT)
            #        httpd.serve_forever()
            #x=threading.Thread(target=http_server)
            #x.start()
            #print("==============================================")
            #print("visualize at http://host ip:{}/{}.html".format(port, self.log_dir))
            #print("==============================================")

    def batch_plot_landmark(self, name, batch_img, dict_label_mark):
        bsize, c, h, w = batch_img.shape
        batch_img = batch_img.detach().cpu().numpy().transpose((0, 2, 3, 1))
        cat_image = np.concatenate(list(batch_img), 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cat_image)
        for label, batch_land in dict_label_mark.items():
            batch_land2d = batch_land[:, :, :2]
            _, n_points, _ = batch_land2d.shape
            batch_land2d[:, :, 1] = h - batch_land2d[:, :, 1]
            batch_land2d = batch_land2d.detach().cpu().numpy()
            offset = np.arange(0, bsize * w, w)
            batch_land2d[:, :, 0] += offset[..., None]
            batch_land2d = batch_land2d.reshape(-1, 2)

            ax.scatter(batch_land2d[:, 0], batch_land2d[:, 1], s=0.5, label=label)
        ax.axis("off")
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)
        fig.savefig(os.path.join(self.plot_dir, '%s.png'%name))
        plt.close()


    def add_scalar(self, name, value, t_iter):
        if not name in self.plot_vals:
            self.plot_vals[name] = [value]
            self.plot_times[name] = [t_iter]
        else:
            self.plot_vals[name].append(value)
            self.plot_times[name].append(t_iter)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.plot_times[name], self.plot_vals[name])
        fig.savefig(os.path.join(self.plot_dir, '%s.png'%name))
        plt.close()
    #add_image('image', torchvision.utils.make_grid(img), num_iter)

    def add_text(self, name, list_text, t_iter, n_word=None):
        if n_word is not None:
            w = n_word*8
        else:
            w = len(list_text[0])*8
        h = len(list_text)*50
        img = Image.fromarray(np.ones((h,w,3), dtype=np.uint8)*255)
        d = ImageDraw.Draw(img)
        for i, text in enumerate(list_text):
            d.text((0, i*50+20), text, fill=(0, 0, 0))
        ww = w*2
        hh = h*2
        img = img.resize((ww, hh))
        img.save(os.path.join(self.plot_dir, "%s.png"%name))

    def add_single_image(self, name, image, t_iter=None):
       image = image.detach().cpu().numpy()
       image = image.transpose((1, 2, 0))
       image = Image.fromarray((image*255).astype(np.uint8))
       image.save(os.path.join(self.plot_dir, "%s.png"%name))

    def add_image(self, name, image, t_iter):
       path_name = os.path.join(self.image_dir, name)
       if not os.path.exists(path_name):
           os.mkdir(path_name)
       image = image.detach().cpu().numpy()
       image = image.transpose((1, 2, 0))
       image = Image.fromarray((image*255).astype(np.uint8))
       image.save(os.path.join(path_name, "%d.png"%t_iter))
       with open(os.path.join(self.log_dir, "image_ticks", name+".txt"), "a") as f:
           f.write(str(t_iter)+'\n')

    def add_single_label(self, name, image, t_iter):
       image = image.detach().cpu().numpy()
       image = Image.fromarray(image.astype(np.uint8)).convert("P")
       image.putpalette(self.palette)
       image.save(os.path.join(self.plot_dir, "%s.png"%name))

    def add_label(self, name, image, t_iter):
       path_name = os.path.join(self.image_dir, name)
       if not os.path.exists(path_name):
           os.mkdir(path_name)
       image = image.detach().cpu().numpy()
       image = Image.fromarray(image.astype(np.uint8)).convert("P")
       image.putpalette(self.palette)
       image.save(os.path.join(path_name, "%d.png"%t_iter))
       with open(os.path.join(self.log_dir, "image_ticks", name+".txt"), "a") as f:
           f.write(str(t_iter)+'\n')

    def write_html_eval(self, base_dir):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_loader = FileSystemLoader(os.path.join(dir_path, "templates"))
        env = Environment(loader=file_loader)
        template = env.get_template('val.html')

        list_exp_ticks = []
        list_exp_paths = []
        list_img_names = []
        list_dirs = os.listdir(base_dir)
        if "mask" in list_dirs:
            list_dirs.remove("mask")
        list_dirs = sorted(list_dirs)
        list_dirs = ["mask"] + list_dirs
        k=0
        for i, exp in enumerate(list_dirs):
            if os.path.isdir(os.path.join(base_dir, exp)) and not exp.startswith("."):
                list_exp_paths.append( os.path.join(exp, "image"))
                list_exp_ticks.append( os.listdir(os.path.join(base_dir, exp, "image")))
                if k == 0:
                    _l = list(filter(lambda x: not x.startswith("."), list_exp_ticks[0]))
                    list_img_names += \
                            os.listdir(
                                os.path.join(base_dir, exp, "image", str(_l[0]))
                                )
                    k += 1
        output = template.render( list_exp_ticks=list_exp_ticks, list_exp_paths=list_exp_paths, list_img_names=list_img_names)
        #print(output)
        with open("{}/validation.html".format(base_dir), "w") as f:
            f.writelines(output)

    def write_console(self, epoch, i, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k,v in self.plot_vals.items():
            #print(v)
            #if v != 0:
            #v = v.mean().float()
            v = v[-1]
            message += '%s: %.4f ' % (k, v)

        print(message)
        prefix = self.log_dir
        with open("{}/logs.txt".format(prefix), "a") as log_file:
            log_file.write('%s\n' % message)

    def write_scalar(self, name, value, t_iter):
        prefix = self.log_dir
        with open("{}/{}.txt".format(prefix, name), "a") as log_file:
            message = '%s %d: %.4f' % (name, t_iter, value)
            print(message)
            log_file.write(message+"\n")


    def write_html(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_loader = FileSystemLoader(os.path.join(dir_path, "templates"))
        env = Environment(loader=file_loader)
        template = env.get_template('train.html')

        prefix = self.log_dir
        re_prefix = self.log_dir.split("/")[-1]
        plotpath = os.path.join(prefix, "plot")
        plotfiles = os.listdir(plotpath)
        plotfiles = list(map(lambda x: os.path.join(re_prefix, "plot", x), plotfiles))

        image_tick_path = []
        imagepath = os.path.join(prefix, "image")
        for folder in os.listdir(imagepath):
            ticks = open("{}/image_ticks/{}.txt".format(prefix, folder), "r").read()
            ticks = ticks.split('\n')[:-1]
            ticks = list(map(lambda x:int(x), ticks))
            folderpath = os.path.join(re_prefix, "image", folder)
            image_tick_path.append({"tick":ticks, "path":folderpath})
        if len(image_tick_path)>0:
            output = template.render( plotfiles=plotfiles, image_tick_path=image_tick_path, title_name=prefix)
        else:
            output = template.render(plotfiles=plotfiles, title_name=prefix)
        #print(output)
        with open("{}.html".format(prefix), "w") as f:
            f.writelines(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Videos to images')
    parser.add_argument('--log_dir', type=str, help='log dir')
    parser.add_argument('--log_dir_eval', type=str, help='log dir')
    args = parser.parse_args()
    logger = Logger(args.log_dir)
    #logger.write_html()
    logger.write_html_eval(args.log_dir_eval)
