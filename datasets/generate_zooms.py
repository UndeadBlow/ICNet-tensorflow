import lensfunpy
import cv2

cam_maker = 'NIKON CORPORATION'
cam_model = 'NIKON D3S'

db = lensfunpy.Database()
cam = db.find_cameras(cam_maker, cam_model)[0]
lens = db.find_lenses(cam)[0]

print(cam)
# Camera(Maker: NIKON CORPORATION; Model: NIKON D3S; Variant: ;
#        Mount: Nikon F AF; Crop Factor: 1.0; Score: 0)

print(lens)
# Lens(Maker: Nikon; Model: Nikkor 28mm f/2.8D AF; Type: RECTILINEAR;
#      Focal: 28.0-28.0; Aperture: 2.79999995232-2.79999995232;
#      Crop factor: 1.0; Score: 110)

focal_length = 1.93
aperture = 2.4
distance = 0
image_path = '/home/undead/Downloads/vlcsnap-2018-06-20-16h58m47s756.jpg'
undistorted_image_path = '/home/undead/Downloads/vlcsnap-2018-06-20-16h58m47s756_2.jpg'

im = cv2.imread(image_path)
height, width = im.shape[0], im.shape[1]

mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
mod.initialize(focal_length, aperture, distance)

undist_coords = mod.apply_geometry_distortion()
im_undistorted = cv2.remap(im, undist_coords, None, cv2.INTER_LANCZOS4)
cv2.imwrite(undistorted_image_path, im_undistorted)