from flask import Flask, make_response
from googleapiclient.errors import HttpError
import numpy as np
import tensorflow as tf
import skimage.transform as skTrans
import nibabel as nib

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
tf.compat.v1.enable_eager_execution()
tf.debugging.set_log_device_placement(False)
app = Flask(__name__)

@tf.function
def get_patches(images, n_patches):
  n, h, w, d = images.shape
  if n is None: n = 1
  # patches_shape = (num of images, num of patches in an img, num of pixels in an patch)
  patches = tf.zeros((n, n_patches[0] * n_patches[1] * n_patches[2],
                     (h//n_patches[0]) * (w//n_patches[1]) * (d//n_patches[2])))

  patch_size = (h//n_patches[0], w//n_patches[1], d//n_patches[2])
  s = n_patches[1] * n_patches[2]

  for ind in range(n):
    for i in range(n_patches[0]):
      for j in range(n_patches[1]):
        for k in range(n_patches[2]):
          patch = images[ind][i * patch_size[0]: (i+1) * patch_size[0], \
                        j * patch_size[1]: (j+1) * patch_size[1], \
                        k * patch_size[2]: (k+1) * patch_size[2] ]

          patch = tf.reshape(patch, [patch_size[0]*patch_size[1]*patch_size[2]])
          patches = tf.tensor_scatter_nd_update(
              patches,indices=[[ind, (i*s) + (j*n_patches[2]) + k]],
              updates= [patch])

  return patches
@tf.function
def get_positions(num_patch, dim):
  result = tf.ones((num_patch, dim))
  for i in range(num_patch):
      for j in range(dim):
        result = tf.tensor_scatter_nd_update(
              result,indices=[[i, j]],
              updates= [np.sin(i / (10000 ** (j / dim))) if j % 2 == 0 \
                        else np.cos(i / (10000 ** ((j - 1) / dim)))])
  return result
def make_prediction(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    image = skTrans.resize(nib.load(image_path).get_fdata()[:,:,28:94],
                         (121,145,10), order=1, preserve_range=True)
    label = np.argmax(model.predict(np.array([image])), axis=1)[0]
    if label == 0:return 'AD'
    if label == 1:return 'CN'
    if label == 2:return 'MCI'
    return label

@app.route('/predict/', methods=['GET'])
def get_record():

    try:
        return make_prediction('model/model_atten_cnn_classfier', 
                               'wc1ADNI_002_S_0295_MR_Axial_PD_T2_FSE__br_raw_20060418201717292_2_S13405_I13724.nii')

    # region Handle Errors
    # No Content, the ID not found
    except KeyError as e:
        response = make_response('', 204)
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Content-Value'] = e.args[0]
        return response
    # Server received the request. but no response is available yet
    # no problem, but Image path and Genome Path not set yet. so can't continue to execute process(run Model)
    except ValueError as e:
        response = make_response('', 102)
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Content-Value'] = e.args[0]
        return response
    # Already Reported, the sample already validated
    except TypeError as e:
        response = make_response('', 208)
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Content-Value'] = e.args[0]
        return response
    except HttpError:
        return make_response("May file not found in this URL or not shareable for everyone", 404)
    except Exception as e:
        return make_response(e.args[0], 400)

