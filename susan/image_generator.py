from PIL import Image, ImageDraw
import random

def generate_random_shapes_bmp(filename, width=640, height=360, shape_count=5, scale_factor=2):
    low_res_width, low_res_height = width // scale_factor, height // scale_factor
    img = Image.new("RGB", (low_res_width, low_res_height), "white")
    draw = ImageDraw.Draw(img)
    
    for _ in range(shape_count):
        shape_type = random.choice(["square", "octagon", "circle", "triangle", "cube"])
        x0, y0 = random.randint(0, low_res_width//1.5), random.randint(0, low_res_height//1.5)
        size = random.randint(10, low_res_width//3)
        x1, y1 = x0 + size, y0 + size
        color = tuple(random.randint(0, 255) for _ in range(3))
        
        if shape_type == "square":
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=None) #"black"
        elif shape_type == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=color, outline=None)
        elif shape_type == "triangle":
            draw.polygon([(x0, y1), ((x0 + x1)//2, y0), (x1, y1)], fill=color, outline=None)
        elif shape_type == "octagon":
            offset = size // 4
            draw.polygon([
                (x0 + offset, y0), (x1 - offset, y0),
                (x1, y0 + offset), (x1, y1 - offset),
                (x1 - offset, y1), (x0 + offset, y1),
                (x0, y1 - offset), (x0, y0 + offset)
            ], fill=color, outline=None)
        elif shape_type == "cube":
            p1 = (x0, y0)
            p2 = (x1, y0)
            p3 = (x1, y1)
            p4 = (x0, y1)
            p5 = (x0 + size//3, y0 - size//3)
            p6 = (x1 + size//3, y0 - size//3)
            p7 = (x1 + size//3, y1 - size//3)
            p8 = (x0 + size//3, y1 - size//3)
            
            # front face
            draw.polygon([p1, p2, p3, p4], fill=color, outline=None)
            # top face
            draw.polygon([p1, p2, p6, p5], fill=color, outline=None)
            # side face
            draw.polygon([p2, p3, p7, p6], fill=color, outline=None)
    
    img = img.resize((width, height), Image.NEAREST)
    
    img.save(filename, "BMP")
    print("Image saved as " + filename)

generate_random_shapes_bmp("random_shapes.bmp")

