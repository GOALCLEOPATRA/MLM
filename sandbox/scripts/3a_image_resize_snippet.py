


from PIL import Image
from resizeimage import resizeimage

data_dir='...'
image_label='...'

# resize
img = Image.open((data_dir+image_label), 'r')
img = resizeimage.resize_contain(img, [128, 128])
img = img.convert('RGB')

# next...
img.save((data_dir+'thumbnail_'+image_label), img.format)