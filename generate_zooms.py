import tensorflow as tf
import numpy as np

def calc_shift(x1, x2, cx, k):
    
    x1 = tf.cast(x1, dtype = tf.float32)
    x2 = tf.cast(x2, dtype = tf.float32)
    x3 = tf.cast(0, dtype = tf.float32)
    
    cx = tf.cast(cx, dtype = tf.float32)
    k = tf.cast(k, dtype = tf.float32)
    res1 = tf.cast(0, dtype = tf.float32)
    res3 = tf.cast(0, dtype = tf.float32)
    
    def calc(x1, x2, x3, cx, k, res1, res3):
        x1, x2 = tf.cond( res3 < 0, lambda : (x3, x2), lambda : (x1, x3) )

        x3 = x1 + (x2 - x1) * 0.5
        res1 = x1 + ((x1 - cx) * k * ((x1 - cx) * (x1 - cx)))
        res3 = x3 + ((x3 - cx) * k * ((x3 - cx) * (x3 - cx)))

        return x1, x2, x3, cx, k, res1, res3

    def check(x1, x2, x3, cx, k, res1, res3):
        thresh = 1.0
        conv = tf.logical_and( tf.greater(-thresh, res1), tf.less(res1, thresh) )
        return conv

    x1, x2, x3, cx, k, res1, res3 = tf.while_loop(check, calc, loop_vars = [x1, x2, x3, cx, k, res1, res3])

    return x1


def getRadialXY(x, y, cx, cy, k, k1, k2, sc, props):

    def scale(x, y, cx, cy, k, k1, k2, sc, props):
        xshift = props[0]
        yshift = props[1]
        xscale = props[2]
        yscale = props[3]

        x = tf.cast(x, dtype = tf.float32)
        x = x * xscale + xshift
        y = tf.cast(y, dtype = tf.float32)
        y = y * yscale + yshift
        r = tf.pow((x - cx), 2) + tf.pow((y - cy), 2)
        y = (y + k * ((y - cy) * r) + k1 * ((y - cy) * tf.pow(r, 2)) + k2 * ((y - cy) * tf.pow(r, 3)))
        x = (x + k * ((x - cx) * r) + k1 * ((x - cx) * tf.pow(r, 2)) + k2 * ((x - cx) * tf.pow(r, 3)))
        return y, x

    def not_scale(x, y, cx, cy, k, k1, k2, sc, props):
        x = tf.cast(x, dtype = tf.float32)
        y = tf.cast(y, dtype = tf.float32)
        r = tf.pow((x - cx), 2.0) + tf.pow((y - cy), 2.0)
        y = (y + k * ((y - cy) * r) + k1 * ((y - cy) * tf.pow(r, 2)) + k2 * ((y - cy) * tf.pow(r, 3)))
        x = (x + k * ((x - cx) * r) + k1 * ((x - cx) * tf.pow(r, 2)) + k2 * ((x - cx) * tf.pow(r, 3)))
        return y, x

    y, x = tf.cond(sc, lambda : scale(x, y, cx, cy, k, k1, k2, sc, props), lambda : not_scale(x, y, cx, cy, k, k1, k2, sc, props))
    return y, x


def fisheye(img, Cx, Cy, k, k1, k2, scale):

    Cx = tf.Print(Cx, [Cx, 'Cx'])
    scale = tf.constant(scale, dtype = tf.bool)
    scale = tf.Print(scale, [scale, 'scale'])

    shape = tf.shape(img)
    shape = tf.Print(shape, [shape, 'shape'])

    xshift = calc_shift(0, Cx - 1, Cx, k)
    xshift = tf.Print(xshift, [xshift, 'xshift'])

    w = tf.cast(shape[0], dtype = tf.float32)
    h = tf.cast(shape[1], dtype = tf.float32)
    w = tf.Print(w, [w, 'w'])
    h = tf.Print(h, [h, 'h'])
    
    newcenterx = w - Cx
    newcenterx = tf.Print(newcenterx, [newcenterx, 'newcenterx'])

    xshift2 = calc_shift(0, newcenterx - 1, newcenterx, k)
    xshift2 = tf.Print(xshift2, [xshift2, 'xshift2'])
    yshift = calc_shift(0, Cy - 1, Cy, k)
    yshift = tf.Print(yshift, [yshift, 'yshift'])
    newcentery = w - Cy
    newcentery = tf.Print(newcentery, [newcentery, 'newcentery'])
    yshift2 = calc_shift(0, newcentery - 1, newcentery, k)
    yshift2 = tf.Print(yshift2, [yshift2, 'yshift2'])
    xscale = (w - xshift - xshift2) / w
    yscale = (h - yshift - yshift2) / h
    xscale = tf.Print(xscale, [xscale, 'xscale'])
    yscale = tf.Print(yscale, [yscale, 'yscale'])

    props = tf.stack([xshift, yshift, xscale, yscale])
    props = tf.Print(props, [props, 'props'])

    y = tf.constant(1, dtype = tf.int32)
    x = tf.constant(1, dtype = tf.int32)

    new_y, new_x = getRadialXY(0, 0, Cx, Cy, k, k1, k2, scale, props)
    map_xy = tf.stack([new_y, new_x])
    map_xy = tf.reshape(map_xy, shape = (1, 2))

    def create_map(x, y, map_xy, Cx = Cx, Cy = Cy, k = k, k1 = k1, k2 = k2, scale = scale, props = props, h = tf.cast(h, dtype = tf.int32)):
        new_y, new_x = getRadialXY(x, y, Cx, Cy, k, k1, k2, scale, props)
        map_xy = tf.concat([map_xy, [[new_y, new_x]]], axis = 0)

        y = y + 1
        x = tf.cond(tf.greater_equal(y, h), lambda : tf.add(x, 1), lambda : x)
        y = tf.cond(tf.greater_equal(y, h), lambda : tf.constant(0), lambda : y)
        x = tf.Print(x, [x, 'loop_x'])
        y = tf.Print(y, [y, 'loop_y'])

        return x, y, map_xy
    
    def check(x, y, map_xy):
        cond = tf.logical_and(tf.less(tf.cast(y, dtype = tf.float32), h), tf.less(tf.cast(x, dtype = tf.float32), w))
        return cond

    x, y, map_xy = tf.while_loop(cond = check, body = create_map, loop_vars = [x, y, map_xy], \
        shape_invariants = [x.get_shape(), y.get_shape(), tf.TensorShape([None, None])], parallel_iterations = 1000,
        back_prop = False, swap_memory = False)

    print(map_xy)
    return map_xy

def get_fisheyed(img, focal_len = 800):
    
    with tf.device('/cpu:0'):
        # parametrs
        f = tf.constant(focal_len, dtype = tf.float32)  # focal'ish
        # magic constants
        f = tf.Print(f, [f, 'f'])
        k = 0.5 * tf.pow(10.0, -2.0) / f
        k1 = k * tf.pow(10.0, -9.0)
        k2 = k1
        k = tf.Print(k, [k, 'k'])

        sz = tf.shape(img)
        sz = tf.Print(sz, [sz, 'image shape'])
        Cx = tf.cast(tf.divide(sz[0], 2), dtype = tf.float32)
        Cx = tf.Print(Cx, [Cx, 'Cx'])
        Cy = tf.cast(tf.divide(sz[1], 2), dtype = tf.float32)
        Cy = tf.Print(Cy, [Cy, 'Cy'])

        fisheyed = fisheye(img, Cx, Cy, k, k1, k2, True)

    return fisheyed


if __name__ == '__main__':

    # test zoom
    filenames = ['/home/undead/Pictures/MVI_2780 11.jpg']
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    images = tf.image.decode_jpeg(value, channels = 3)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print('inited')
        #print('images:', sess.run(images).shape)
        mapxy = get_fisheyed(images)
        print(sess.run(mapxy))