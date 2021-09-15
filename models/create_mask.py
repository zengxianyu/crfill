import cv2
import numpy as np
import random
from PIL import Image, ImageDraw
import os
import pdb
import math

class MaskCreator:
    def __init__(self, list_mask_path=None, base_mask_path=None, match_size=False):
        self.match_size = match_size
        if list_mask_path is not None:
            filenames = open(list_mask_path).readlines()
            msk_filenames = list(map(lambda x: os.path.join(base_mask_path, x.strip('\n')), filenames))
            self.msk_filenames = msk_filenames
        else:
            self.msk_filenames = None


    def object_shadow(self, h, w, blur_kernel=7, noise_loc=0.5, noise_range=0.05):
        """
        img: rgb numpy
        return: rgb numpy
        """
        mask = self.object_mask(h, w)
        kernel = np.ones((blur_kernel+3,blur_kernel+3),np.float32)
        expand_mask = cv2.dilate(mask,kernel,iterations = 1)
        noise = np.random.normal(noise_loc, noise_range, mask.shape)
        noise[noise>1] = 1
        mask = mask*noise
        mask = mask + (mask==0)
        kernel = np.ones((blur_kernel,blur_kernel),np.float32)/(blur_kernel*blur_kernel)
        mask = cv2.filter2D(mask,-1,kernel)
        return mask, expand_mask


    def object_mask(self, image_height=256, image_width=256):
        if self.msk_filenames is None:
            raise NotImplementedError
        hb, wb = image_height, image_width
        # object mask as hole
        mask = Image.open(random.choice(self.msk_filenames))
        ## randomly resize
        wm, hm = mask.size
        if self.match_size:
            r = float(min(hb, wb)) / max(wm, hm)
            r = r /2
        else:
            r = 1
        scale = random.gauss(r, 0.5)
        scale = scale if scale > 0.5 else 0.5
        scale = scale if scale < 2 else 2.0
        wm, hm = int(wm*scale), int(hm*scale)
        mask = mask.resize((wm, hm))
        mask = np.array(mask)
        mask = (mask>0)
        if mask.sum() > 0:
            ## crop object region
            col_nz = mask.sum(0)
            row_nz = mask.sum(1)
            col_nz = np.where(col_nz!=0)[0]
            left = col_nz[0]
            right = col_nz[-1]
            row_nz = np.where(row_nz!=0)[0]
            top = row_nz[0]
            bot = row_nz[-1]
            mask = mask[top:bot, left:right]
        else:
            return self.object_mask(image_height, image_width)
        ## place in a random location on the extended canvas
        hm, wm = mask.shape
        canvas = np.zeros((hm+hb, wm+wb))
        y = random.randint(0, hb-1)
        x = random.randint(0, wb-1)
        canvas[y:y+hm, x:x+wm] = mask
        hole = canvas[int(hm/2):int(hm/2)+hb, int(wm/2):int(wm/2)+wb]
        th = 100 if self.match_size else 1000
        if hole.sum() < hb*wb / th:
            return self.object_mask(image_height, image_width)
        else:
            return hole.astype(np.float)

    def rectangle_mask(self, image_height=256, image_width=256, min_hole_size=64, max_hole_size=128):
        mask = np.zeros((image_height, image_width))
        hole_size = random.randint(min_hole_size, max_hole_size)
        hole_size = min(int(image_width*0.8), int(image_height*0.8), hole_size)
        x = random.randint(0, image_width-hole_size-1)
        y = random.randint(0, image_height-hole_size-1)
        mask[x:x+hole_size, y:y+hole_size] = 1
        return mask

    def random_brush(
            self,
            max_tries,
            image_height=256,
            image_width=256,
            min_num_vertex = 4,
            max_num_vertex = 18,
            mean_angle = 2*math.pi / 5,
            angle_range = 2*math.pi / 15,
            min_width = 12,
            max_width = 48):
        H, W = image_height, image_width
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)
        for _ in range(np.random.randint(max_tries)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)
            if np.random.random() > 0.5:
                mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.uint8)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 0)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 1)
        return mask

    def random_mask(self, image_height=256, image_width=256, hole_range=[0,1]):
        coef = min(hole_range[0] + hole_range[1], 1.0)
        #mask = self.random_brush(int(20 * coef), image_height, image_width)
        while True:
            mask = np.ones((image_height, image_width), np.uint8)
            def Fill(max_size):
                w, h = np.random.randint(max_size), np.random.randint(max_size)
                ww, hh = w // 2, h // 2
                x, y = np.random.randint(-ww, image_width - w + ww), np.random.randint(-hh, image_height - h + hh)
                mask[max(y, 0): min(y + h, image_height), max(x, 0): min(x + w, image_width)] = 0
            def MultiFill(max_tries, max_size):
                for _ in range(np.random.randint(max_tries)):
                    Fill(max_size)
            MultiFill(int(10 * coef), max(image_height, image_width) // 2)
            MultiFill(int(5 * coef), max(image_height, image_width))
            mask = np.logical_and(mask, 1 - self.random_brush(int(20 * coef), image_height, image_width))
            hole_ratio = 1 - np.mean(mask)
            if hole_ratio >= hole_range[0] and hole_ratio <= hole_range[1]:
                break
        return 1-mask

    def stroke_mask(self, image_height=256, image_width=256, max_vertex=5, max_mask=5, max_length=128):
        max_angle = np.pi
        max_brush_width = max(1, int(max_length*0.4))
        min_brush_width = max(1, int(max_length*0.1))

        mask = np.zeros((image_height, image_width))
        for k in range(random.randint(1, max_mask)):
            num_vertex = random.randint(1, max_vertex)
            start_x = random.randint(0, image_width-1)
            start_y = random.randint(0, image_height-1)
            for i in range(num_vertex):
                angle = random.uniform(0, max_angle)
                if i % 2 == 0:
                    angle = 2*np.pi - angle
                length = random.uniform(0, max_length)
                brush_width = random.randint(min_brush_width, max_brush_width)
                end_x = min(int(start_x + length * np.cos(angle)), image_width)
                end_y = min(int(start_y + length * np.sin(angle)), image_height)
                mask = cv2.line(mask, (start_x, start_y), (end_x, end_y), color=1, thickness=brush_width)
                start_x, start_y = end_x, end_y
                mask = cv2.circle(mask, (start_x, start_y), int(brush_width/2), 1)
            if random.randint(0, 1):
                mask = mask[:, ::-1].copy()
            if random.randint(0, 1):
                mask = mask[::-1, :].copy()
        return mask


def get_spatial_discount(mask):
    H, W = mask.shape
    shift_up = np.zeros((H, W))
    shift_up[:-1, :] = mask[1:, :]
    shift_left = np.zeros((H, W))
    shift_left[:, :-1] = mask[:, 1:]

    boundary_y = mask - shift_up
    boundary_x = mask - shift_left
    
    boundary_y = np.abs(boundary_y)
    boundary_x = np.abs(boundary_x)
    boundary = boundary_x + boundary_y
    boundary[boundary != 0 ] = 1
#     plt.imshow(boundary)
#     plt.show()
    
    xx, yy = np.meshgrid(range(W), range(H))
    bd_x = xx[boundary==1]
    bd_y = yy[boundary==1]
    dis_x = xx[..., None] - bd_x[None, None, ...]
    dis_y = yy[..., None] - bd_y[None, None, ...]
    dis = np.sqrt(dis_x*dis_x + dis_y*dis_y)
    min_dis = dis.min(2)
    gamma = 0.9
    discount_map = (gamma**min_dis)*mask
    return discount_map




if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import pdb
    mask_creator = MaskCreator()
    mask = mask_creator.random_mask(image_height=512, image_width=512)
    Image.fromarray((mask*255).astype(np.uint8)).save("output/mask.png")
