#!/opt/local/bin/python

# longitude_ = np.zeros(300,dtype=np.int)
# latitude_  = np.zeros(300,dtype=np.int)
# 
# lon_shape = longitude_.shape[0]
# lat_shape = latitude_.shape[0]
# 
# n0 = 10
# n1 = 10
# 
# irange0 = [i/n0 for i in range(lat_shape)]
# irange1 = [j/n1 for j in range(lon_shape)]
# i01,j01 = np.meshgrid(irange0,irange1)
# iTile = i01 + (1+max(irange0))*j01

from scipy.spatial import KDTree
print('starting...')


# x,y = np.mgrid[0:5,2:8]
x,y = np.mgrid[0:100,0:100]
z   = np.zeros((100,100))
x_r = x.ravel()
y_r = y.ravel()
xy = zip(x_r,y_r)

# tree = KDTree(list(xy))

# pts = np.array([[0,0], [2.1,2.9]])
# tree.query(pts)

hull=ConvexHull(xy)
xv=x_r[hull.vertices]
yv=y_r[hull.vertices]
xbar = sum(xv)/float(len(xv))
ybar = sum(yv)/float(len(yv))

dx = xv-xbar
dy = yv-ybar
dx2 = dx*dx
dy2 = dy*dy

d2 = dx2 + dy2
d  = np.sqrt(d2)
r  = np.min(d)
r = r * 0.5

dx = x-xbar
dy = y-ybar
dx2 = dx*dx
dy2 = dy*dy
d2  = dx2 + dy2
d   = np.sqrt(d2)

zInD = z[np.where(d <= r)]
print('z.shape: ',z.shape,'zInD.shape: ',zInD.shape)



print('done...')

