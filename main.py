
from matplotlib.pyplot import get_current_fig_manager
import skimage.io as skio
import skimage.draw as draw
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import subprocess

from scipy.spatial import Delaunay
from os import listdir
from os.path import isfile, join

nick = skio.imread("img/nick.png")
chad = skio.imread("img/gigachad.png")

def show(img):
    skio.imshow(img)
    skio.show()

def save(img, name):
    skio.imsave(name, img)

def jpg_name(name):
    return "img/" + name + ".jpg"

def png_name(name):
    return "img/" + name + ".png"

def population_name(name):
    return "img/population/" + name + ".jpg"

def read_png(name, astype = "float64"):
    png = skio.imread(png_name(name))
    if len(png.shape) == 2:
        # Handle the grayscale case
        return png.astype(astype)
    # Remove the alpha channel.
    return png[:,:,:3].astype(astype)

def read_jpg(name, astype = "float64"):
    jpg = skio.imread(jpg_name(name))
    if len(jpg.shape) == 2:
        # Handle the grayscale case
        return jpg.astype(astype)
    return jpg[:,:,:3].astype(astype)
    
def points_name(name):
    return "points/" + name + ".points"

def select_points(image, num_points, name):
    """ Choose NUM_POINTS on IMAGE and save them as NAME. """
    
    points = []
    plt.imshow(image)
    points = plt.ginput(num_points, 0)
    points.extend([(0, 0), (image.shape[1], 0), (0, image.shape[0]), (image.shape[1], image.shape[0])])

    plt.close()
    pickle_name = re.split("\.", name)[0] + ".points"
    pickle.dump(points, open("points/" + pickle_name, "wb"))
    return points

def retrieve_points(name):
    """ Retrieve points for NAME from the points directory. """
    points = pickle.load(open(points_name(name), "rb"))
    return np.array(points)

def retrieve_points_path(path):
    """ Retrieve points for NAME from the points directory. """
    points = pickle.load(open(path, "rb"))
    return np.array(points)

def display_points(imageA_name, imageB_name):
    """ Display selected points over two images. """
    pointsA = retrieve_points(imageA_name)
    pointsB = retrieve_points(imageB_name)
    point_display_name_A = jpg_name(re.split("\.", imageA_name)[-1] + "points")
    point_display_name_B = jpg_name(re.split("\.", imageB_name)[-1] + "points")

    imgA = skio.imread(png_name(imageA_name))
    plt.imshow(imgA)
    for i, point in enumerate(pointsA):
        plt.scatter(point[0], point[1], color="red")
        plt.annotate(str(i), (point[0], point[1]))
    plt.savefig(point_display_name_A)
    plt.close()

    imgB = skio.imread(png_name(imageB_name))
    plt.imshow(imgB)
    for point in pointsB:
        plt.scatter(point[0] - 1, point[1] - 2, color="pink")
    plt.savefig(point_display_name_B)
    plt.close()

    point_display_A = skio.imread(point_display_name_A)
    point_display_B = skio.imread(point_display_name_B)
    skio.imshow(np.concatenate([point_display_A, point_display_B]))
    skio.show()

def display_population_points(name):
    """ Display the points of an image from the dataset folder rather than the base img folder. """
    # Read from the POPULATION folder rather than the base folder.
    # pointsA = retrieve_points(name)
    pointsA = parse_points(name + ".pts")

    # Read from the POPULATION folder rather than the base folder.
    imgA = skio.imread("img/population/" + name + ".jpg")

    # Use the population mean in place:
    # imgA = skio.imread("img/population_mean.jpg")
    
    plt.imshow(imgA)
    for i, point in enumerate(pointsA):
        plt.scatter(point[0], point[1], color="red")
        plt.annotate(str(i), (point[0], point[1]))
    plt.savefig(jpg_name(re.split("\.", name)[-1] + "points"))
    plt.close()

def interpolate(a, b, mix):
    """ 
    Interpolate between point A & point B by MIX.
    A MIX of 0 is purely point A, while a MIX of 1 is purely point B.
    """
    return a + ((b - a) * mix)

def triangles(points, show = False):
    """
    Given a set of points, generate a set of triangles connecting the points.

    A triangle is defined as

              0   1
        [   
    0       [x1, y1]
    1       [x2, y2]
    2       [x3, y3]
                        ]

    where (triangle[0][0], triangle[0][1]) make up the x, y coordinates 
    of the first point.

    SHOW displays the selected triangles; the code for SHOW is courtesy of
    the SciPy documentation.

    Returns the sets of points that make up each triangle in the image as
    
    [
        [vertex1, vertex2, vertex3]         (triangle 1)
        [vertex1, vertex2, vertex3]         (triangle 2)
                    ...
        [vertex1, vertex2, vertex3]         (triangle n)
                                      ]
    
    a list of length-3 lists of indices into the POINTS array.
    """

    tri = Delaunay(points)

    if show:
        plt.triplot(points[:,0], max(points[:,1]) - points[:,1], tri.simplices)
        plt.plot(points[:,0], max(points[:,1]) - points[:,1], 'o')
        plt.show()

    return tri.simplices

def computeAffine(triangle, target):
    """ 
    The affine matrix transformation happens here! 
    Translates from TRIANGLE to TARGET.
    """
    A = np.matrix([triangle[:,0], triangle[:, 1], [1, 1, 1]])
    B = np.matrix([target[:,0], target[:, 1], [1, 1, 1]])
    return B * np.linalg.inv(A)

def apply_masked_affine(mask, image, src_tri, target_tri, population=True):
    """ 
    Generates an image the same size as IMAGE, inserts the pixels surrounded by
    SRC_TRI warped into the shape of TARGET_TRI, and discards the rest.
    Returns the triangular set of pixels in the correct coordinates.
    """
    A = computeAffine(src_tri, target_tri)

    final_mask = mask.copy()
    final_mask[0] = mask[0]
    final_mask[1] = mask[1]

    affined = (np.linalg.inv(A) * final_mask).astype(np.int)
    cc, rr, _ = affined
    
    # If, by chance, mask indices are out of range, try clipping these values. 
    #cc = np.clip(cc, 0, image.shape[0])
    #rr = np.clip(rr, 0, image.shape[1])

    final_mask = final_mask.astype(np.int)
    canvas = np.zeros_like(image)

    canvas[final_mask[1], final_mask[0]] = image[rr, cc]
    return canvas

def triangle_bool_matrix(triangle, image_shape):
    """ Masks the triangle of the image we want to add in to a result. """
    tri_buf = triangle
    if len(image_shape) == 2:
        shape = (image_shape[1], image_shape[0], 1)
    else:
        shape = (image_shape[1], image_shape[0], image_shape[2])
    points = draw.polygon(tri_buf[:,0], tri_buf[:,1], shape=shape)
    return np.vstack([points, np.ones(len(points[0]))])

def transform(i_am, become, jpg = False, population = False):
    """ 
        "I am become death, destroyer of worlds."
    Become death, the destroyer of worlds, or something else by
    making yourself or some source object take the shape of another.

    Inputs:
        I_AM - the base name of a source image & its corresponding files.
        BECOME - the base name of a "destination" image, points, etc.
        (JPG) - is SRC a JPG?
        (POPULATION) - does SRC come from the FEI dataset?
    
    Returns:
        You, as the destroyer of worlds (an image).
    
    Retreive the set of points belonging to I_AM and BECOME.
    Then, for each triangular region defined by I_AM, create a transformation
    matrix to map that region to the corresponding one in BECOME.
    Stretch all of the points/colors within each defined triangle to encompass
    BECOME, and add all the results into a single image.

    """

    # Retrieve actual images referenced by the names
    if population:
        src = skio.imread(population_name(i_am))
    else:
        if jpg:
            src = read_jpg(i_am, astype="uint8")
        else:
            src = read_png(i_am)
    #dst = read_png(become)

    # Retrieve serialized points for each image
    if population:
        # Read & build up the points from the dataset.
        src_points = parse_points(i_am + ".pts")
        src_points.extend([(0, 0), (249, 0), (0, 299), (249, 299)])
    else:
        src_points = retrieve_points(i_am)
    dst_points = retrieve_points(become)

    # Generate triangles based on the points defined in both images.
    src_simplices = triangles(dst_points, False)
    
    # Create a blank canvas to paint our image on.
    result = np.zeros_like(src)

    # For each triangle,
    for i in range(len(src_simplices)):
        
        # Gather the vertices specified by this triangle.
        #
        # The below is not a typo; we want to use SRC's simplices for both
        # images to ensure that each triangle refers to the same points,
        # specifically because TRIANGLES can return totally different
        # edges even with similar points.
        src_tri = np.array(src_points)[src_simplices[i]]
        dst_tri = np.array(dst_points)[src_simplices[i]]

        # Mask out the SRC triangle and apply the matrix that 
        # transforms it into the DST triangle.
        target_mask = triangle_bool_matrix(dst_tri, src.shape)
        transformed_tri = apply_masked_affine(target_mask, src, src_tri, dst_tri)

        result += transformed_tri

    return result

def linear(frame_count):
    """ Steady increase from zero to one. """
    return [i / len(frame_count) for i in range(len(frame_count))]

def linear_extrapolate(frame_count):
    """ Steady increase from negative one to two - extrapolating by two times as far. """
    return [((3 * i) - 1) / len(frame_count) for i in range(len(frame_count))]

def easeInOutExpo(x):
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 0.5:
        return pow(2, 10 * x - 5) / 2
    else:
        return (2 - pow(2, -10 * x + 5)) / 2

def ease_in_out(frame_count):
    """ Slow to start off, a quick transition in the middle, and a slow finish. """
    length = len(frame_count)
    return [easeInOutExpo(i / length) for i in range(length)]

def midway(frame_count):
    """ Ignore all parameters & just compute the midway image. """
    return [0.5 for _ in range(len(frame_count))]

def mix(mix_value):
    """ A list of just the MIX_VALUE passed in. """
    return [mix_value for _ in range(len(mix_value))]

def transform_frames(i_am, become, frames=60, interp_func=ease_in_out):
    """ 
    Generate a sequence of images that progressively morph two objects.

    Inputs:
        I_AM - the base name of a source image & its corresponding files.
        BECOME - the base name of a "destination" image, points, etc.
        (FRAMES) - number of frames of output.

    Returns:
        Nothing (to save memory for large FRAME counts).
        As a side effect, saves output frames in /out/[I_AM]/

    Tip: pass in frames = 1 & interp_func = midway to get a midway image.
    """

    # Retrieve actual images referenced by the names
    src = read_png(i_am)
    dst = read_png(become)

    if src[0,0,:].max(axis=0) > 1:
        src = src / 255

    if dst[0,0,:].max(axis=0) > 1:
        dst = dst / 255

    # Retrieve serialized points for each image
    src_points = retrieve_points(i_am)
    dst_points = retrieve_points(become)

    # Generate triangles based on the points defined in both images.
    src_simplices = triangles(src_points, False)

    # Alternatively, uncomment the next line to go in the reverse direction.
    # src_simplices = triangles(dst_points, False)
    
    frame = 0
    for mix in interp_func(range(frames)):
        print(mix)
        # Create a blank canvas to paint our image on.
        result = np.zeros_like(src)

        # For each triangle,
        for i in range(len(src_simplices)):
            
            # Gather the vertices specified by this triangle.
            #
            # The below is not a typo; we want to use SRC's simplices for both
            # images to ensure that each triangle refers to the same points,
            # specifically because TRIANGLES can return totally different
            # edges even with similar points.
            src_tri = src_points[src_simplices[i]]
            dst_tri = dst_points[src_simplices[i]]
            
            interp_tri = (src_tri * (1 - mix)) + (dst_tri * mix)

            # Mask out the SRC triangle and apply the matrix that 
            # transforms it into the DST triangle.
            target_mask = triangle_bool_matrix(interp_tri, dst.shape)
            transformed_src_tri = apply_masked_affine(target_mask, src, src_tri, interp_tri)
            transformed_dst_tri = apply_masked_affine(target_mask, dst, dst_tri, interp_tri)

            result += ((1 - mix) * transformed_src_tri) + (mix * transformed_dst_tri)

        save(result, "out/" + i_am + "/" + str(frame) + ".jpg")
        frame += 1

    return

def transform_frames_many(sequence, frames=60, interp_func=ease_in_out):
    """ 
    Generate a sequence of images that progressively morph SEVERAL objects.
    """

    frame = 0
    for i in range(len(sequence) - 1):

        # Retrieve actual images referenced by the names
        src = read_png(sequence[i])
        dst = read_png(sequence[i + 1])

        if len(src.shape) == 2:
            src = np.dstack([src, src, src])
        
        if src[src.shape[0] // 2, src.shape[1] // 2, :].max(axis=0) > 1:
            src = src / 255

        if len(dst.shape) == 2:
            dst = np.dstack([dst, dst, dst])

        if dst[src.shape[0] // 2, src.shape[1] // 2, :].max(axis=0) > 1:
            dst = dst / 255

        # Retrieve serialized points for each image
        src_points = retrieve_points(sequence[i])
        dst_points = retrieve_points(sequence[i + 1])

        # Generate triangles based on the points defined in both images.
        src_simplices = triangles(src_points, False)

        # Alternatively, uncomment the next line to go in the reverse direction.
        # src_simplices = triangles(dst_points, False)
        
        for mix in interp_func(range(frames)):
            print(mix)
            # Create a blank canvas to paint our image on.
            result = np.zeros_like(src)

            # For each triangle,
            for i in range(len(src_simplices)):
                
                # Gather the vertices specified by this triangle.
                #
                # The below is not a typo; we want to use SRC's simplices for both
                # images to ensure that each triangle refers to the same points,
                # specifically because TRIANGLES can return totally different
                # edges even with similar points.
                src_tri = src_points[src_simplices[i]]
                dst_tri = dst_points[src_simplices[i]]
                
                interp_tri = (src_tri * (1 - mix)) + (dst_tri * mix)

                # Mask out the SRC triangle and apply the matrix that 
                # transforms it into the DST triangle.
                target_mask = triangle_bool_matrix(interp_tri, dst.shape)
                transformed_src_tri = apply_masked_affine(target_mask, src, src_tri, interp_tri)
                transformed_dst_tri = apply_masked_affine(target_mask, dst, dst_tri, interp_tri)

                result += ((1 - mix) * transformed_src_tri) + (mix * transformed_dst_tri)

            save(result, "out/" + sequence[0] + "/" + str(frame) + ".jpg")
            frame += 1

    return

def midway_image(src, dst):
    """ Calculates the midway image between SRC and DST. """
    transform_frames(src, dst, frames = 1, interp_func = midway)

#midway_image("fan", "brandon_fan")

def population_average():
    dst_points, individual_points = read_population_points()
    pickle.dump(dst_points, open("points/population_average.points", "wb"))

    # Path to the population's images.
    path = "./img/population/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    dst_simplices = triangles(dst_points, False)

    result = np.zeros_like(skio.imread("./img/population/" + files[0])).astype("float64")
    #result = np.zeros((300, 350))

    for j, file in enumerate(files):
        src = skio.imread("./img/population/" + file)
        src_points = individual_points[j]
        #if not (j + 12) % 80:
        #    show(result)
        print(src.shape)

        for i in range(len(dst_simplices)):
            
            src_tri = np.array(src_points)[dst_simplices[i]]
            dst_tri = np.array(dst_points)[dst_simplices[i]]

            # Mask out the SRC triangle and apply the matrix that 
            # transforms it into the DST triangle.
            target_mask = triangle_bool_matrix(dst_tri, src.shape)
            #show(target_mask)
            transformed_src_tri = apply_masked_affine(target_mask, src, src_tri, dst_tri)

            result += ((1 / len(files)) * transformed_src_tri)
            #result += (transformed_src_tri).astype('uint8')
    
    return result.astype('uint8')

def read_population_points():
    """ Return both the average face's points and the individual faces' parsed points. """
    path = "./points/population/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    #print("Reading ", files)

    all_points = []
    average_point_x = [0] * 46
    average_point_y = [0] * 46

    for file in files:
        tuples = parse_points(file)
        for i in range(len(tuples)):
            average_point_x[i] += tuples[i][0]
            average_point_y[i] += tuples[i][1]
        tuples.extend([(0, 0), (249, 0), (0, 299), (249, 299)])
        all_points.append(tuples)
        
    average_points = [(average_point_x[i] / len(files), average_point_y[i] / len(files)) for i in range(46)]
    average_points.extend([(0, 0), (249, 0), (0, 299), (249, 299)])

    return average_points, all_points

def parse_points(file):
    """ 
    Parse an FEI face's points.
    
    The file is structured as
    #   version: 1               #
    #   n_points: [QUANTITY]     #
    #   {                        #
    #      [X] [Y]               #   1
    #        .                   #    .
    #         .                  #     .
    #          .                 #      .
    #      [X] [Y]               #   n_points
    #   }                        #
    #                            #

    Return the x & y coordinates of each point in a list of tuples.
    """
    file = open("./points/population/" + file, "r")
    lines = file.readlines()

    # Extract the number of points from line 2.
    # They should all have 46, but just to be sure:
    point_count = int(lines[1].split(" ")[1])
    point_tuples = []
    for line in range(3, 3 + point_count):
        # Split on the space between the two numbers.
        # The first index should be the x coordinate,
        #            and the second the y coordinate.
        points = lines[line].split(" ")

        point_tuples.append((float(points[0]), float(points[1])))
    return point_tuples

def select_nick_gigachad_points():
    select_points(nick, 36, "nick")
    select_points(chad, 36, "gigachad")

#select_nick_gigachad_points()
#display_points("gigachad", "nick")
#display_points("brandon_fan", "fan")

#display_population_points("1a")
#display_points("nick_population", "nick1")
#display_population_points("population_mean")

def points_to_video(src_name, dst_name):
    """ Chain starting from image generation all the way to an output mp4."""
    transform_frames(src_name, dst_name, 360)
    cmd = ['ffmpeg', '-r', '60', '-i', './out/' + src_name + '/%d.jpg', '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2", 'output.mp4']
    retcode = subprocess.call(cmd)
    if not retcode == 0:
        raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))

def generate_video(src_name, dst_name, points = 16):
    """ Chain starting from specifying images and how many points to add to generating a full video. """
    src = read_png(src_name, "uint8")
    dst = read_png(dst_name, "uint8")
    select_points(src, points, src_name)
    select_points(dst, points, dst_name)
    
    points_to_video(src_name, dst_name)

def generate_gigachad_video():
    """ Hardcodes the process of creating the original Nick-to-Gigachad video. """
    transform_frames("nick", "gigachad", 360)
    cmd = ['ffmpeg', '-r', '60', '-i', './out/nick/%d.jpg', '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2", 'output.mp4']
    retcode = subprocess.call(cmd)
    if not retcode == 0:
        raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))

###############################################################
# Demos

# Try generating a morph video by selecting your own points
# generate_video("brandon_fan", "fan", 24)

# Use the provided points to recreate the same result as in the website.
# generate_gigachad_video()

def nick_as_population_average():
    """ Compute the population average, then show my face as the population average. """
    population_average()
    show(transform("nick_population", "population_mean", jpg=True))

nick_as_population_average()

def morph_dataset():
    """ Morph a number of dataset images to the dataset average. """
    show(population_average())
    show(transform("4a", "population_mean", population = True))
    show(transform("16b", "population_mean", population = True))
    show(transform("61a", "population_mean", population = True))
    show(transform("61b", "population_mean", population = True))
    show(transform("161a", "population_mean", population = True))
    show(transform("149b", "population_mean", population = True))
    show(transform("194a", "population_mean", population = True))
    show(transform("186a", "population_mean", population = True))

# Extrapolate!
# transform_frames("gigachad", "nick", 6, interp_func=linear_extrapolate)

def generate_bodybuilder_video(select_points = False):
    """ Bell & whistle. """
    # Music video
    bodybuilders = ["scott", "oliva", "schwarzeneggar", "zane", "coleman", "cutler", "scott"]
    if select_points:
        for bodybuilder in bodybuilders:
            img = read_png(bodybuilder)
            if len(img.shape) == 2:
                img = np.dstack([img, img, img])
            if max(img[0,:, 0]) > 1:
                img = img / 255
            select_points(img, 12, bodybuilder)
    transform_frames_many(bodybuilders, frames=120)
    cmd = ['ffmpeg', '-r', '60', '-i', './out/scott/%d.jpg', '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2", 'bodybuilder.mp4']
    retcode = subprocess.call(cmd)
    if not retcode == 0:
        raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))

#generate_bodybuilder_video()
