from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw

class Landingfield:

    def __init__(self, position, direction, dimension):
        self.dir = np.array(direction) # flight direction
        self.dim = np.array(dimension) # length, width of touchdown zone
        self.pos = np.array(position) # Middel of Brick One

    def is_inside(self, p):
        length, width = self.dim
        p1, p2, p3, p4 = self.get_rectangle()
        dist_length = np.dot ((p3-p2)/np.linalg.norm(p3-p2), p3-p)
        inside_length = (0<=dist_length<=length)
        dist_width = np.dot ((p2-p1)/np.linalg.norm(p2-p1), p2-p)
        inside_width = (0<=dist_width<=width)
        #print('pos={} p={}  p3-p2={} p3-p={} d_len={:.2f} d_width={:.2f} inside={}{}'.format(self.pos, p, p3-p2, p3-p, dist_length, dist_width,inside_length, inside_width))
        return inside_length and inside_width

    def get_rectangle(self):
        length, width = self.dim
        dir90 = np.array([self.dir[1], -self.dir[0]])
        p1 = self.pos - dir90 * width/2
        p2 = self.pos + dir90 * width/2
        p3 = p2 + self.dir * length
        p4 = p3 - dir90 * width
        return p1, p2, p3, p4

def draw_poly(data, coordinates, color):
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(p) for p in coordinates], fill=color)
    return np.asarray(img)

dir = np.random.uniform(0.1,1,2)* np.random.choice([-1,1], 2)
dir = dir/np.linalg.norm(dir)
dim =np.array([900,50])
pos = np.random.uniform(900,1100,2)
lf = Landingfield(pos, dir, dim)
xs = [x[0] for x in lf.get_rectangle()]
xs.append(xs[0])
ys = [x[1] for x in lf.get_rectangle()]
ys.append(ys[0])
print('direction={}'.format(lf.dir))
#plt.plot (xs, ys, 'r-')
data = np.ones((2000,2000))*50
new_data = draw_poly(data, lf.get_rectangle(), 200)
plt.imshow(data)
plt.plot(lf.pos[0], lf.pos[1], 'r.')
plt.ion()
plt.show()
# input()
# p1, p2, p3, p4 = lf.get_rectangle()
# plt.plot(p1[0],p1[1],'r.')
# input()
# plt.plot(p2[0],p2[1],'r.')
# input()
# plt.plot(p3[0],p3[1],'r.')
# input()
# plt.plot(p4[0],p4[1],'r.')
# input()
#plt.xlim(-1000,1000)
#plt.ylim(-1000,1000)
#for i in range(0,2000):
#    p = np.random.uniform(-1000,1000,2)
#    if lf.is_inside(p):
#        plt.plot(p[0], p[1],'g.')
#    else:
#        plt.plot(p[0], p[1],'b.')
    #input()
plt.show()
input()
