import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf

W = 848
H = 480

# We'll start the camera by specifying the type of stream and its resolution:

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Search for rgb camera and stop run if it is not connected
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Stream activation
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

print("[INFO] Starting streaming...")
profile = pipeline.start(config)
print("[INFO] Camera ready.")

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1.5 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
point_cloud = rs.pointcloud()

print("[INFO] Loading model...")
PATH_TO_CKPT = "frozen_inference_graph.pb"

# We start by creating Graph object and loading it from file:

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

# Initialize the relevant input and output vectors needed in this sample:

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

colors_hash = {}

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        frames = align.process(frames)
        # Get aligned frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        scaled_size = (color_frame.width, color_frame.height)

        points = point_cloud.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image_expanded = np.expand_dims(bg_removed, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                    feed_dict={image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        print("[INFO] drawing bounding box on detected objects...")
        print("[INFO] each detected object has a unique color")

        for idx in range(int(num)):
            class_ = classes[idx]
            score = scores[idx]
            box = boxes[idx]
            print(" [DEBUG] class : ", class_, "idx : ", idx, "num : ", num)
            
            if class_ not in colors_hash:
                colors_hash[class_] = tuple(np.random.choice(range(256), size=3))
            
            if score > 0.6:
                left = int(box[1] * color_frame.width)
                top = int(box[0] * color_frame.height)
                right = int(box[3] * color_frame.width)
                bottom = int(box[2] * color_frame.height)
                
                width = right - left
                height = bottom - top
                bbox = (int(left), int(top), int(width), int(height))
                p1 = (left, top)
                p2 = (right, bottom)
                # draw box
                r, g, b = colors_hash[class_]
                cv2.rectangle(bg_removed, p1, p2, (int(r), int(g), int(b)), 2, 1)

                # x,y,z of bounding box
                obj_points = verts[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])].reshape(-1, 3)
                zs = obj_points[:, 2]

                z = np.median(zs)

                ys = obj_points[:, 1]
                ys = np.delete(ys, np.where(
                    (zs < z - 1) | (zs > z + 1)))  # take only y for close z to prevent including background

                my = np.amin(ys, initial=1)
                My = np.amax(ys, initial=-1)

                height = (My - my)  # add next to rectangle print of height using cv library
                height = float("{:.2f}".format(height))
                print("[INFO] object height is: ", height, "[m]")
                height_txt = str(height) + "[m]"

                # Write some Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (p1[0], p1[1] + 20)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2
                cv2.putText(bg_removed, height_txt,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Image classification in depth', bg_removed)
        cv2.imshow('RealSense', bg_removed)
        
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    print("[INFO] stop streaming ...")
    pipeline.stop()