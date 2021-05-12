from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from deep_glide.sim import Runway

def draw_poly(data, coordinates, color):
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(p) for p in coordinates], fill=color)
    return np.asarray(img)

dir = np.random.uniform(0.1,1,2)* np.random.choice([-1,1], 2)
dir = dir/np.linalg.norm(dir)
dim =np.array([900,300])
pos = np.random.uniform(900,1100,2)
lf = Runway(pos, dir, dim)
xs_rect = [x[0] for x in lf.get_rectangle()]
xs_rect.append(xs_rect[0])
ys_rect = [x[1] for x in lf.get_rectangle()]
ys_rect.append(ys_rect[0])
xs_arr = [x[0] for x in lf.arrow]
ys_arr = [x[1] for x in lf.arrow]

print('direction={}'.format(lf.dir))
plt.plot (xs_rect, ys_rect, 'r-')
plt.plot (xs_arr, ys_arr, 'r-')

data = np.ones((2000,2000))*50
new_data = draw_poly(data, lf.get_rectangle(), 200)
plt.imshow(data)
plt.plot(lf.pos[0], lf.pos[1], 'r.')
plt.ion()
plt.show()
#plt.xlim(-1000,1000)
#plt.ylim(-1000,1000)
for i in range(0,2000):
    p = np.random.uniform(0,2000,2)
    if lf.is_inside(p):
        plt.plot(p[0], p[1],'g.')
    else:
        plt.plot(p[0], p[1],'b.')
plt.show()
input()
