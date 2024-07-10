import cv2
import numpy as np
import math
from retinaface import RetinaFace

def crop_image(image, rectangle):                                                       #Helper funkcija za croppat sliko
    min_x = rectangle[0][0][0]
    max_y = rectangle[0][0][1]
    max_x = rectangle[0][1][0]
    min_y = rectangle[0][2][1]
    cropped_image = image[min_y:max_y, min_x:max_x]
    return cropped_image

def smaller_rectangle(rectangle, scale_left, scale_right, scale_top, scale_bottom):     #Helper funkcija za croppat pravkotnik v formatu: [min_x, min_y, max_y, max_y]
    min_x = rectangle[0]
    max_y = rectangle[1]
    max_x = rectangle[2]
    min_y = rectangle[3]
    
    centroid_x = (min_x + max_x) / 2
    centroid_y = (min_y + max_y) / 2
    
    leftmost = int(centroid_x - ((centroid_x - min_x) * scale_left))
    rightmost = int(((max_x - centroid_x) * scale_right) + centroid_x)
    topmost = int(((max_y - centroid_y) * scale_top) + centroid_y)
    bottommost = int(centroid_y - ((centroid_y - min_y) * scale_bottom))
    
    return [[(leftmost, topmost), (rightmost, topmost), (rightmost, bottommost), (leftmost, bottommost)]]


def create_text_bounds(image, id):                                                      #Funkcija, ki nam poišče mejne pravokotnike okoli besedila
    greyImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(greyImg, (7,7), 0)
    thers = cv2.threshold(blurImg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilateImg = cv2.dilate(thers, kernel, iterations=2)
    
    vrni = []
    
    face = detect_face(image)                                                           #Detekcija obraza
    if face:
        search_rectangle = scale_search_rectangle(image, face, 0.10, 0.1175)            #Definiramo iskalni pravokotnik okrog obraza
        face = scale_search_rectangle(image, face, 0.04, 0.07)                          #Povečamo območje obraza
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0,0,255), 1)
        vrni.append([face[0], face[1], face[2] - face[0], face[3] - face[1]])
        cv2.imwrite(f"{id}_face.jpg", image)
        min_x, min_y, max_x, max_y = face
        sides = [
            [min_x, min_y, max_x, min_y],  # Top side
            [max_x, min_y, max_x, max_y],  # Right side
            [max_x, max_y, min_x, max_y],  # Bottom side
            [min_x, max_y, min_x, min_y]   # Left side
        ]
        cv2.imwrite(f"{id}_face.jpg", image)
    else:
        print("Face not detected!")
        
    contours = cv2.findContours(dilateImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    
    counter = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        counter += 1
        if w >= 10 and h >= 10 and compare_face_and_box(sides, [[x,y,w,h]], image, counter):
            vrni.append([x,y,w,h])
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    output_path = f"{id}_boxes.jpg"
    cv2.imwrite(output_path, image)
    return vrni, search_rectangle

def scale_search_rectangle(image, face, scale_hor, scale_ver):                          #Helper funkcja za razširitev pravokotnika v formatu: [min_x, min_y, max_y, max_y]
    min_x, min_y, max_x, max_y = face
    height, width, channels = image.shape
    scale_hor = scale_hor * width
    scale_ver = scale_ver * height
    
    if min_x - scale_hor > 0:
        min_x =  min_x - scale_hor
    else:
        min_x = 0
        
    if max_x + scale_hor < width:
        max_x =  max_x + scale_hor
    else:
        max_x = width    
        
    if min_y - scale_ver > 0:
        min_y =  min_y - scale_ver 
    else:
        min_y = 0
        
    if max_y + scale_ver < height:
        max_y =  max_y + scale_ver
    else:
        max_y = height
        
    return int(min_x), int(min_y), int(max_x), int(max_y)


def compare_face_and_box(sides, box, image, counter):                                   #Funkcija, ki odstrani mejne pravokotnike vsebovane v obraznem pravokotniku
    nice_image = np.copy(image)
    min_x = sides[0][0]
    min_y = sides[0][1]
    max_x = sides[0][2]
    max_y = sides[1][3]
    bx, by, bw, bh = box[0]
    if min_x >= bx and min_y >= by and max_x <= bx + bw and max_y <= by + bh:
        nice_image = cv2.rectangle(nice_image, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (255,0,0), 1)
        return False
    else:
        for line in sides:
            cv2.line(nice_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
            if does_line_touch_boxes(line, box, image):
                nice_image = cv2.rectangle(nice_image, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (255,0,0), 1)
                return False
            else:
                nice_image = cv2.rectangle(nice_image, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (0,255,0), 1)
        cv2.imwrite("check_fail.jpg", nice_image)
    return True
    
    
  
# Code snippet below (function detect_lines) obtained from Stackoverflow
# Source: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
# Author: Ahi
# Accessed: 10.05.2024  

def detect_lines(image, length_threshold, id):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(9, 9),1.1)
    low_threshold = 110  #prej 130
    high_threshold = 130 #prej 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 100  # angular resolution in radians of the Hough grid
    threshold = 70  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 0.5 * length_threshold  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save the image with lines
    output_path = f"{id}_lines.jpg"
    cv2.imwrite(output_path, line_image)
    
    return lines

def line_length(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Code snippet below (function onSegment, orientation, doIntersect) obtained from GeeksforGeeks
# Source: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# Accessed: 10.05.2024

def onSegment(p, q, r): 
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))): 
        return True
    return False

def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Collinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  
      
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
    if (val > 0): 
        # Clockwise orientation 
        return 1
    elif (val < 0): 
          
        # Counterclockwise orientation 
        return 2
    else: 
          
        # Collinear orientation 
        return 0
    
    
def doIntersect(p1,q1,p2,q2): 
      
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # Special Cases 
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
  
    # If none of the cases 
    return False


def are_endpoints_outside_boxes(line, bounding_boxes):                             #Funkcija, ki določa ali je črta izven mejnih pravokotnikov
    for box in bounding_boxes:
        bx, by, bw, bh = box
        if (line[0] < bx or line[0] > bx + bw or line[1] < by or line[1] > by + bh) and (line[2] < bx or line[2] > bx + bw or line[3] < by or line[3] > by + bh):
            continue
        else:
            return False
    return True

def does_line_touch_boxes(line, bounding_boxes, image, one=False):                #Funkcija, ki določa ali se črta dotika mejnih pravokotnikov
    for box in bounding_boxes:
        if one:
            x = box[0]
            y = box[1]
            w = box[2] - x
            h = box[3] - y
        else:
            x, y, w, h = box
        # Calculate line equations for each side of the bounding box
        sides = [
            [x, y, x + w, y],  # Top side
            [x + w, y, x + w, y + h],  # Right side
            [x, y + h, x + w, y + h],  # Bottom side
            [x, y, x, y + h]  # Left side
        ]
        # Check if line intersects with any side of the bounding box
        for side in sides:
            if doIntersect((line[0], line[1]), (line[2], line[3]), (side[0], side[1]), (side[2], side[3])):
                return True
    return False
    

def compare_lines_and_boxes(lines, bounding_boxes, image, search_rectangle, id):            #Funkcija, ki določa katere črte ustrezajo našim pogojem
    total_length = 0
    count_lines = 0
    cv2.rectangle(image, (search_rectangle[0], search_rectangle[1]), (search_rectangle[2], search_rectangle[3]), (0,0,255), 2)
    for line in lines:
        line = line[0]
        if are_endpoint_inside_box(line, search_rectangle) and not does_line_touch_boxes(line, [search_rectangle], image, True) and not does_line_touch_boxes(line, bounding_boxes, image) and are_endpoints_outside_boxes(line, bounding_boxes):
            height, width, channels = image.shape
            count_lines += 1
            total_length += line_length(line)
            cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)

    cv2.imwrite(f"{id}_final.jpg", image)
    return total_length

def are_endpoint_inside_box(line, box):                                                     #Funkcija, ki določa ali je črta v celoti znotraj pravokotnika
    x1, y1, x2, y2 = line
    x_min, y_min, x_max, y_max = box
    
    def is_point_inside_box(x, y, x_min, y_min, x_max, y_max):
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    return is_point_inside_box(x1, y1, x_min, y_min, x_max, y_max) and is_point_inside_box(x2, y2, x_min, y_min, x_max, y_max)


def detect_face(image):                                                                    #Wrapper funkcija za detekcijo obraza z Retina Face
    resp = RetinaFace.detect_faces(image)
    if resp:
        return resp['face_1']['facial_area']
    else:
        return None
    
def scale_face_rectangle(rectangle, scale_left, scale_right, scale_bottom, scale_top):          #Helper funkcja za razširitev pravokotnika v formatu: [min_x, min_y, width, height]
    min_x = rectangle[0]
    min_y = rectangle[1]
    max_y = min_y + rectangle[3]
    max_x = min_x + rectangle[2]
    
    centroid_x = (max_x - min_x) / 2
    centroid_y = (max_y - min_y) / 2
    
    left = int(centroid_x - ((centroid_x - min_x) * scale_left))
    right = int(((max_x - centroid_x) * scale_right) + centroid_x)
    top = int(((max_y - centroid_y) * scale_top) + centroid_y)
    bottom = int(centroid_y - ((centroid_y - min_y) * scale_bottom))
    
    return [left, bottom, right - min_x, top - min_y]


def process_image(image_path, id):                                                              #Main funkcija metode, za input dobi pot do slike in številko za lažje sledenje izpisom
    length = 0
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    length_thres = 0.1 * min(height, width)
    image_rectangle = [0, height, width, 0]
    smaller = smaller_rectangle(image_rectangle, 0.95, 0.95, 0.95, 0.95)                        #Najprej se slika nekoliko obreže
    cropped_image = crop_image(image, smaller)
    copied_image = cropped_image.copy()                                                         
    copied2_image = cropped_image.copy()
    bound_boxes, search_rectangle = create_text_bounds(cropped_image, id)                       #Poiščemo mejne pravokotnike okrog besedila in obraza
    lines = detect_lines(copied_image, length_thres, id)                                        #Poiščemo ravne črte na sliki
    if lines is not None and bound_boxes is not None:
        length = compare_lines_and_boxes(lines, bound_boxes, copied2_image, search_rectangle, id)       #Primerjamo najdene črte z najdenimi pravokotniki
    else:
        length = 0
        length_thres = 100
        
    if length >= length_thres:
        return 1
    else:
        return 0
