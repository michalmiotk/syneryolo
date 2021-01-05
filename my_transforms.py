import torchvision


class Transform_img_labels():
    def __init__(self):
        self.size = 448
        self.grid_size=7
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize(
            (self.size, self.size)), torchvision.transforms.ToTensor()])
        self.class_dict = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'dog', 'horse', 'motorbike', 'person', 'sheep', 'sofa', 'diningtable', 'pottedplant', 'train', 'tvmonitor']

        self.cell_size = self.size/self.grid_size

    def cell_calc(self, xmin, ymin, xmax, ymax):
        xcenter = (xmin+xmax)/2
        ycenter = (ymin+ymax)/2
        xcell, ycell = int(xcenter/self.cell_size), int(ycenter/self.cell_size)
        return ycell, xcell
    
    def width_height_percent(self, xmin, ymin, xmax, ymax):
        return (ymax-ymin)/448, (xmax-xmin)/448

    def cell_offset_percent(self, xmin, ymin, xmax, ymax):
        ycell, xcell = self.cell_calc( xmin, ymin, xmax, ymax)
        ycenter, xcenter = ymax-ymin, xmax-xmin
        y_offset_percent, x_offset_percent = (ycenter-ycell*self.cell_size)/self.cell_size, (xcenter-xcell*self.cell_size)/self.cell_size
        return y_offset_percent, x_offset_percent

    def __call__(self, img, target):
        label = target
        shape_x = int(label['annotation']['size']['width'])
        shape_y = int(label['annotation']['size']['height'])
        coef_x = shape_x/self.size
        coef_y = shape_y/self.size
        self.factor = (1/shape_x, 1/shape_y)
        objects = label['annotation']['object']
		
        if not isinstance(objects, list):
            objects = list([objects])
		
        tr_objects = []
        for obj in list(objects):
            
            xmin = float(obj['bndbox']['xmin'])*coef_x
            ymin = float(obj['bndbox']['ymin'])*coef_y
            xmax = float(obj['bndbox']['xmax'])*coef_x
            ymax = float(obj['bndbox']['ymax'])*coef_y  
            class_obj = self.class_dict.index(obj['name'])
            cell = self.cell_calc(xmin,ymin, xmax,ymax)
            width_height = self.width_height_percent(xmin,ymin, xmax,ymax)
            cell_offset = self.cell_offset_percent(xmin,ymin, xmax,ymax)

            tr_objects.append([class_obj, cell, width_height, cell_offset])

        return self.transform(img), tr_objects
    
