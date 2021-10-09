from utils import get_data
from PIL import Image, ImageDraw, ImageFont

def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    # IMPLEMENT THIS FUNCTION
    
    # color class mapping
    colors = {1 : 'red', 2: 'yellow', 3: 'blue' }
    
    for pic in range(len(ground_truth)):
        
        # load image
        im = Image.open('./data/Images/' + ground_truth[pic]['filename'])
        #im.show()
    
        # create object to draw on it
        draw = ImageDraw.Draw(im)
        
        for box in range(len(ground_truth[pic]['boxes'])):
            
            # the boxes need to be reshaped to adapt it to the PIL Image coordinate system
            corners = ground_truth[pic]['boxes'][box]
            corners_new = [corners[1], corners[0], corners[3], corners[2] ]
            
            # draw a rectangle on it
            draw.rectangle(corners_new, 
                           outline = colors[ground_truth[pic]['classes'][box]], 
                           width=3)
            
        im.show()
            
        im.save(ground_truth[pic]['filename'])

if __name__ == "__main__": 
    ground_truth, _ = get_data()
    viz(ground_truth)