import pymongo 
import pprint
import json

def is_inside(bbox1,bbox2,b=5):
  x1,y1,w1,h1=bbox1
  x2,y2,w2,h2=bbox2
  if (x1+b)>x2 and (y1+b)>y2 and (x1+w1)<(x2+w2) and (y1+h1)<(y2+h2) :
    return True
  if x1>x2 and y1>y2 and (x1+w1)<(x2+w2+b) and (y1+h1)<(y2+h2+b) :
    return True
  return False

def associate_widgets(y):
  widgets=[widget for widget in y if widget['class']!='text']
  labels=[widget for widget in y if widget['class']=='text']    
  for widget in widgets:
    for label in labels:
      if is_inside(label['box'],widget['box']):
        try:
          y.remove(label)
        except:
          print(label)
          print(widget)
        if 'text' in widget.keys():
          widget['text']+=label['text']
        else:
          widget['text']=label['text']
  return y

def overlap(bbox1,bbox2):
  x1,y1,w1,h1=bbox1
  x2,y2,w2,h2=bbox2
  x_left = max(x1,x2)
  x_right = min(x1+w1,x2+w2)
  y_top = max(y1,y2)
  y_bottom = min(y1+h1,y2+h2)
  if y_top>y_bottom:
    return False
  else:
    return max(0,x_right-x_left)

def make_heights(widgets,d=30):

  # modifying text heights
  labels=[widget for widget in widgets if widget['class']=='text']
  if labels:
    h=int(sum([x['box'][3] for x in labels])/len(labels))
    for label in labels:
      label['box'][3]=h
  
  # modifying input box heights
  input_boxes=[widget for widget in widgets if widget['class'] in ['drop_down','text_box','date']]
  if input_boxes:
    h=int(sum([x['box'][3] for x in input_boxes])/len(input_boxes))
    for input_box in input_boxes:
      input_box['box'][3]=h
    
  # modifying check input
  checks=[widget for widget in widgets if widget['class'] in ['check_box','radio']]
  if checks:
    h=int(sum([x['box'][3] for x in checks])/len(checks))
    for check in checks:
      check['box'][3]=h
  
  # make heights
  widgets=sorted(widgets,key=lambda x: x['box'][1])
  for i in range(1,len(widgets)):
    if abs(widgets[i]['box'][1] -widgets[i-1]['box'][1])<d:
      widgets[i]['box'][1]=widgets[i-1]['box'][1]
  
  # justify
  widgets=sorted(widgets,key=lambda x: x['box'][0])
  for i in range(1,len(widgets)):
    if abs(widgets[i]['box'][0] -widgets[i-1]['box'][0])<2*d:
      widgets[i]['box'][0]=widgets[i-1]['box'][0]

  return sorted(widgets,key=lambda x: (x['box'][1],x['box'][0]))

def filter_widgets(widgets):
  # text overlap
  labels=[widget for widget in widgets if widget['class']=='text']
  for label1 in labels:
    if label1 in labels:
      for label2 in labels:
        x1,y1,w1,h1=label1['box']
        x2,y2,w2,h2=label2['box']
        if label1!=label2 and overlap(label1['box'],label2['box']) > min(w1,w2)/10:
          print("overlapping",label1,label2,"of",overlap(label1['box'],label2['box']))
          labels.remove(label2)
          widgets.remove(label2)
          label1['box'][0]=min(x1,x2)
          label1['box'][1]=min(y1,y2)
          label1['box'][2]=max(x1+w1,x2+w2)-label1['box'][0]
          label1['box'][3]=max(y1+h1,y2+h2)-label1['box'][1]

  components=[widget for widget in widgets if widget['class']!='text']
  for comp1 in components:
    if comp1 in components:
      for comp2 in components:
        x1,y1,w1,h1=comp1['box']
        x2,y2,w2,h2=comp2['box']
        if comp1!=comp2 and overlap(comp1['box'],comp2['box']) > min(w1,w2)/2:
          print("overlapping",comp1,comp2,"of",overlap(label1['box'],label2['box']))
          if comp1['box'][2]>comp2['box'][2]:
            components.remove(comp2)
            widgets.remove(comp2)
          else:
            components.remove(comp1)
            widgets.remove(comp1)
            break
  return widgets

def resize(img):
  w,h,_=img.shape
  if max(w,h)<1000:
    return img
  ratio=1000/max(w,h)
  nh=int(h*ratio)
  nw=int(w*ratio)
  img = cv2.resize(img, (nh,nw), interpolation = cv2.INTER_AREA)
  return img

import time
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
PATH_TO_SAVED_MODEL = "./models/form_model/saved_model"
PATH_TO_CFG = "./models/form_model/pipeline.config"
PATH_TO_LABELS = "./models/form_model/label_map.txt"

print('Loading model... ', end='')
start_time = time.time()


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']

detection_model = model_builder.build(model_config=model_config, is_training=False)

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
import cv2
import tensorflow as tf
def form_widget_recognition(image,path=True):
  if path:
    image_np = cv2.imread(image)
  else:
    image_np=image
  
  img_h,img_w,_=image_np.shape

  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image_np)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]

  # input_tensor = np.expand_dims(image_np, 0)
  
  detections = detect_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  image_np_with_detections = image_np.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)
  
  output=[]
  min_score_thresh=.30
  for box,class_id,score in zip(detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores']):
    if score >= min_score_thresh:
      output.append({
          'box': box,
          'box': [ int(box[1]*img_w),
                  int( box[0]*img_h),
                  int((box[3]-box[1])*img_w),
                  int((box[2]-box[0])*img_h)],
          'class': category_index[class_id]['name'],
          'score': float(score)
      })
  cv2.imwrite("./result.jpg",image_np_with_detections)
  print('Done')
  return output

client = pymongo.MongoClient("mongodb+srv://pavan:pavan@cluster0.srvng.mongodb.net/form_recognition_app?retryWrites=true&w=majority")
db=client.form_recognition_app
print(db.list_collection_names())
User=db.User
Form=db.Form


# from flask_ngrok import run_with_ngrok
import codecs
from flask import Flask, request, jsonify, send_from_directory, make_response
import base64
import io
from PIL import Image
from flask_cors import CORS, cross_origin
import json
import os
from bson import ObjectId


app = Flask(__name__)
# run_with_ngrok(app)   #starts ngrok when the app is run
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/",methods=['GET'])
@cross_origin()
def greeting():
  return json.dumps({'message':"Hello World!"}),200

@app.route("/design",methods=['POST'])
@cross_origin()
def design():
    user_id = request.form.get('user_id')
    name = request.form.get('form_name')
    image = request.files['image']
    image.save('./input.jpg')
    image_np=cv2.imread("input.jpg")
    image_np=resize(image_np)
    cv2.imwrite('./input.jpg',image_np)
    form_widgets=form_widget_recognition(image_np,False)
    form_widgets=filter_widgets(form_widgets)
    with open("./label_boxes.txt","w+") as f:
      f.write('\n'.join([' '.join(map(str, label['box'])) for label in form_widgets if label['class']=='text']))
    os.system("python3 ./ocr_script.py")
    with open("./labels.txt","r") as f:
      for label,line in zip([w for w in form_widgets if w['class']=='text'],f.readlines()):
        label['text']=line.strip()
    form_widgets=associate_widgets(form_widgets)
    form_widgets=make_heights(form_widgets)
    h,w,_=image_np.shape
    result={'w':w,'h':h,'widgets':form_widgets}
    form_record={
        'user_id':user_id,
        'form_name':name,
        'form_meta_data':result
    }
    id=Form.insert_one(form_record)
    form_record['form_id']=str(id.inserted_id)
    form_record.pop('_id')
    return json.dumps(form_record),200


@app.route("/signin",methods=['POST'])
@cross_origin()
def signin():
    email = request.form.get('email')
    password = request.form.get('password')
    
    user=User.find_one({'email':email,'password':password})
    if user:
      user.pop('password')
      user['id']=str(user.pop('_id'))
      return json.dumps(user),200
    else:
      return "Email or Password doesn't exist!",409


@app.route("/signup",methods=['POST'])
@cross_origin()
def signup():
    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')
    print(name,email,password)
    if User.count_documents({"email":email})>0:
      return 'User name already exists!',409
    
    if User.count_documents({"name":name})>0:
      return 'Email already exists!',409
    
    id=User.insert({'email':email,'name':name,'password':password})
    return json.dumps({'id':str(id)}),200


@app.route("/user-forms/<id>",methods=['GET'])
@cross_origin()
def getUserForms(id):
    forms=[]
    for form in Form.find({'user_id':id}):
      form['form_id']=str(form.pop('_id'))
      forms.append(form)
    forms.sort(key=lambda x: x.get('form_name'),reverse=True)
    return json.dumps(forms),200


@app.route("/user-forms/<id>",methods=['PUT'])
@cross_origin()
def updateForm(id):
    form_record=json.loads(request.form.get('form'))
    if Form.find_one({'_id':ObjectId(id)})==None:
      return "Form doesn't exist", 404
    updateRes=Form.update_one({'_id':ObjectId(id)},{"$set":form_record})
    if updateRes.matched_count<=0:
      return "Couldn't update form!",409
    return "Form Updated!",200


@app.route("/user-forms/<id>",methods=['DELETE'])
@cross_origin()
def deleteForm(id):
    if Form.find_one({'_id':ObjectId(id)})==None:
      return "Form doesn't exist", 404
    deleteRes=Form.delete_one({'_id':ObjectId(id)})
    if deleteRes.deleted_count<=0:
      return "Couldn't delete form!",409
    return "Form Deleted!",200

if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(host = '0.0.0.0', port = port)
