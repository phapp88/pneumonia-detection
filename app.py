import boto3
import io
import math
import numpy as np
import os
import plotly.graph_objs as go
import psycopg2
import random
import streamlit as st

from dotenv import load_dotenv
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from sklearn.cluster import KMeans

load_dotenv()

@st.cache(allow_output_mutation=True, hash_funcs={'_thread.RLock': lambda _: None})
def init_connection():
  if 'DYNO' in os.environ:
    return psycopg2.connect(os.environ['DATABASE_URL'], sslmode='require')

  return psycopg2.connect(
    host=os.environ['DB_HOST'],
    port=os.environ['DB_PORT'],
    dbname=os.environ['DB_NAME'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASSWORD']
  )

def login(connection):
  with connection:
    with connection.cursor() as cursor:
      select_statement = 'SELECT * FROM users WHERE username = %s AND password = %s'
      params = (st.session_state.username, st.session_state.password)
      cursor.execute(select_statement, params)
      results = cursor.fetchall()

      if len(results) == 1:
        st.session_state.is_logged_in = True

      else:
        st.session_state.login_message = 'The username or password you entered is incorrect.'

@st.cache()
def get_model():
  boto3.client('s3').download_file('pneumonia-detection3', 'model.h5', 'model.h5')
  return load_model('model.h5')

def preprocess_image(image):
  if image.mode != 'RGB':
    image = image.convert('RGB')
  
  image = image.resize((299, 299))
  image = img_to_array(image)
  image = np.expand_dims(image, axis = 0)
  image = preprocess_input(image)

  return image

def get_diagnosis(model, image):
  preprocessed_image = preprocess_image(image)
  prediction = model.predict(preprocessed_image)[0][0]
  diagnosis = 'Normal' if prediction < 0.5 else 'Pneumonia'

  return diagnosis

def set_image_to_diagnose(image):
  st.session_state.image_to_diagnose = image

def diagnosis_page(model):
  st.image(st.session_state.image_to_diagnose)
  diagnosis = get_diagnosis(model, st.session_state.image_to_diagnose)
  st.write('Diagnosis: ' + diagnosis)
  st.button(label='Back to dashboard', on_click=set_image_to_diagnose, args=(None, ))

def login_page(connection):
  with st.form(key='login_form', clear_on_submit=True):
    st.header('Login')

    st.text_input(label='Username', key='username')
    st.text_input(label='Password', type='password', key='password')

    st.write(st.session_state.login_message)

    st.form_submit_button(label='Sign In', on_click=login, args=(connection, ))

def pca_scatterplot_page():
  features = np.load('features.npy')

  n_clusters = 2
  kmeans = KMeans(n_clusters = n_clusters)
  kmeans.fit(features)

  layout = go.Layout(
    title = '<b>Cluster Visualization</b>',
    yaxis = { 'title': '<i>Y</i>' },
    xaxis = { 'title': '<i>X</i>' }
  )

  colors = ['red', 'green', 'blue']
  trace = [go.Scatter3d() for _ in range(n_clusters)]
  for i in range(n_clusters):
    my_members = (kmeans.labels_ == i)
    index = [h for h, g, in enumerate(my_members) if g]
    trace[i] = go.Scatter3d(
      x = features[my_members, 0],
      y = features[my_members, 1],
      z = features[my_members, 2],
      mode = 'markers',
      marker = { 'size': 2, 'color': colors[i] },
      hovertext = index,
      name = 'Cluster' + str(i)
    )
  
  fig = go.Figure(data = [trace[0], trace[1]], layout = layout)

  st.plotly_chart(fig)

def upload_image_page():
  st.header('Upload an Image')
  st.file_uploader(label='Choose a file', type = ['png', 'jpg', 'jpeg'], key='uploaded_file')

def view_images_page():
  s3_bucket = boto3.resource('s3').Bucket('pneumonia-detection3')

  all_image_keys = [obj.key for obj in s3_bucket.objects.filter(Prefix='xrays/')]

  sample_image_keys = random.sample(all_image_keys, 8)
  sample_images = [Image.open(s3_bucket.Object(image_key).get()['Body']) for image_key in sample_image_keys]

  num_columns = 4
  num_rows = math.ceil(len(sample_images) / num_columns)

  for i in range(num_rows):
    columns = st.columns(num_columns)

    for j in range(len(columns)):
      column = columns[j]

      with column:
        image = sample_images[i * num_columns + j]
        resized_image = image.resize((160, 160))
        st.image(resized_image)
        st.button(label='Get diagnosis', key = str((i, j)), on_click=set_image_to_diagnose, args=(image, ))

def main():
  connection = init_connection()
  model = get_model()

  if not st.session_state:
    st.session_state.is_logged_in = False
    st.session_state.login_message = ''
    st.session_state.uploaded_file = None
    st.session_state.image_to_diagnose = None

  if not st.session_state.is_logged_in:
    login_page(connection)
  
  elif st.session_state.image_to_diagnose is not None:
    diagnosis_page(model)

  elif st.session_state.uploaded_file is not None:
    st.session_state.image_to_diagnose = Image.open(io.BytesIO(st.session_state.uploaded_file.getvalue()))
    diagnosis_page(model)
  
  else:
    dashboard_page = st.sidebar.selectbox('Choose a page', ['Upload Image', 'View Images', 'PCA Scatterplot'])

    if dashboard_page == 'Upload Image':
      upload_image_page()
    
    elif dashboard_page == 'View Images':
      view_images_page()
    
    elif dashboard_page == 'PCA Scatterplot':
      pca_scatterplot_page()
  
if __name__ == '__main__':
  main()
