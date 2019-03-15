import argparse
import io
import time
import traceback

import numpy as np
import tensorflow as tf
from adain.coral import coral
from adain.image import load_image, prepare_image
from adain.nn import build_vgg, build_decoder
from adain.norm import adain
from adain.weights import open_weights
from scipy.misc import imsave


def save_image_in_memory(image, data_format='channels_first'):
    if data_format == 'channels_first':
        image = np.transpose(image, [1, 2, 0])  # CHW --> HWC
    image *= 255
    image = np.clip(image, 0, 255)
    imgByteArr = io.BytesIO()
    imsave(imgByteArr, image.astype(np.uint8), 'JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


vgg = None
encoder = None
decoder = None
target = None
weighted_target = None
image = None
content = None
style = None
persistent_session = None
data_format = 'channels_first'


def parse_args():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer')

    parser.add_argument('--content', help='File path to the content image',
                        default='/tf-adain/input/my_style.jpg')

    parser.add_argument('--style', default='/tf-adain/input/my_content.jpg',
                        help="""File path to the style image,
            or multiple style images separated by commas if you want to do style
            interpolation or spatial control""")

    parser.add_argument('--content_size', default=512,
                        type=int, help="""Maximum size for the content image, keeping
            the original size if set to 0""")

    parser.add_argument('--style_size', default=512, type=int,
                        help="""Maximum size for the style image, keeping the original
            size if set to 0""")

    parser.add_argument('--crop', action='store_true', help="""If set, center
            crop both content and style image before processing""")

    parser.add_argument('--save_ext', default='jpg',
                        help='The extension name of the output image')

    parser.add_argument('--gpu', default=0, type=int,
                        help='Zero-indexed ID of the GPU to use; for CPU mode set to -1')

    parser.add_argument('--output_dir', default='/tf-adain/my_output/',
                        help='Directory to save the output image(s)')

    parser.add_argument('--preserve_color', action='store_true',
                        help='If set, preserve color of the content image')

    parser.add_argument('--alpha', default=1.0, type=float,
                        help="""The weight that controls the degree of stylization. Should be
            between 0 and 1""")

    args = parser.parse_args()
    return args


def _build_graph(vgg_weights, decoder_weights, alpha, data_format):
    if data_format == 'channels_first':
        image = tf.placeholder(shape=(None, 3, None, None), dtype=tf.float32)
        content = tf.placeholder(shape=(1, 512, None, None), dtype=tf.float32)
        style = tf.placeholder(shape=(1, 512, None, None), dtype=tf.float32)
    else:
        image = tf.placeholder(shape=(None, None, None, 3), dtype=tf.float32)
        content = tf.placeholder(shape=(1, None, None, 512), dtype=tf.float32)
        style = tf.placeholder(shape=(1, None, None, 512), dtype=tf.float32)

    target = adain(content, style, data_format=data_format)
    weighted_target = target * alpha + (1 - alpha) * content

    with open_weights(vgg_weights) as w:
        vgg = build_vgg(image, w, data_format=data_format)
        encoder = vgg['conv4_1']

    if decoder_weights:
        with open_weights(decoder_weights) as w:
            decoder = build_decoder(weighted_target, w, trainable=False,
                                    data_format=data_format)
    else:
        decoder = build_decoder(weighted_target, None, trainable=False,
                                data_format=data_format)

    return image, content, style, target, encoder, decoder


def initialize_model():
    global args
    global vgg
    global encoder
    global decoder
    global target
    global weighted_target
    global image
    global content
    global style
    global persistent_session
    global data_format
    alpha = 1.0

    args = parse_args()

    graph = tf.Graph()
    # build the detection model graph from the saved model protobuf
    with graph.as_default():
        image = tf.placeholder(shape=(None, 3, None, None), dtype=tf.float32)
        content = tf.placeholder(shape=(1, 512, None, None), dtype=tf.float32)
        style = tf.placeholder(shape=(1, 512, None, None), dtype=tf.float32)

        target = adain(content, style, data_format=data_format)
        weighted_target = target * alpha + (1 - alpha) * content

        with open_weights('models/vgg19_weights_normalized.h5') as w:
            vgg = build_vgg(image, w, data_format=data_format)
            encoder = vgg['conv4_1']

        with open_weights('models/decoder_weights.h5') as w:
            decoder = build_decoder(weighted_target, w, trainable=False, data_format=data_format)

        # the default session behavior is to consume the entire GPU RAM during inference!
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.12

        # the persistent session across function calls exposed to external code interfaces
        persistent_session = tf.Session(graph=graph, config=config)

        persistent_session.run(tf.global_variables_initializer())

    print('Initialized model')


def infer(inputs_dict):
    global data_format

    # only update the negative fields if we reach the end of the function - then update successfully
    result_data = {"content-type": 'text/plain',
                   "data": None,
                   "success": False,
                   "error": None}

    try:
        print('Starting inference')
        start = time.time()

        content_image = load_image(io.BytesIO(inputs_dict['content']), args.content_size, args.crop)
        style_image = load_image(io.BytesIO(inputs_dict['style']), args.style_size, args.crop)

        if args.preserve_color:
            style_image = coral(style_image, content_image)
        style_image = prepare_image(style_image)
        content_image = prepare_image(content_image)
        style_feature = persistent_session.run(encoder, feed_dict={
            image: style_image[np.newaxis, :]
        })
        content_feature = persistent_session.run(encoder, feed_dict={
            image: content_image[np.newaxis, :]
        })
        target_feature = persistent_session.run(target, feed_dict={
            content: content_feature,
            style: style_feature
        })

        output = persistent_session.run(decoder, feed_dict={
            content: content_feature,
            target: target_feature
        })

        output_img_bytes = save_image_in_memory(output[0], data_format=data_format)

        result_data["content-type"] = 'image/jpeg'
        result_data["data"] = output_img_bytes
        result_data["success"] = True
        result_data["error"] = None

        print('Finished inference and it took ' + str(time.time() - start))
        return result_data


    except Exception as err:
        traceback.print_exc()
        result_data["error"] = traceback.format_exc()
        return result_data
