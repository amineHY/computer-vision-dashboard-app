import streamlit as st 
import cv2 as cv
import numpy as np
import urllib
import os

# Display settings for bouunding box (bbox)
colorWhite = (255, 255, 255)
colorBlack = (0, 0, 0)
colorRed = (255, 0, 0)
colorGreen = (0, 255, 0)
colorBlue = (0, 0, 255)
fontFace = cv.FONT_HERSHEY_SIMPLEX
thickText = 1


def UI():

    list_of_apps = [
                'Empty',
                'Object Detection',
                'Face Detection',
                'Fire Detection']
    guiParam = {}

    ####################################################
    # configure the main page
    ####################################################
    st.image("./media/logo_inveesion.png","my logo", width=50)
    st.title('Computer Vision Dashboard')


    ####################################################
    # configure the sidebar
    ####################################################
    st.sidebar.markdown("### :arrow_right: Settings")

    dataSource = st.sidebar.radio(
            'Please select the source of your Image Apllication', ['Image: Demo', 'Image: Upload', 'Image: URL'])


    # Select the application from the UI
    selectedApp = st.sidebar.selectbox(
        'Chose an AI Application', list_of_apps)

    if selectedApp == 'Empty':
        st.sidebar.warning('Select an application from the list')

    st.header(' :arrow_right: Application: {}'.format(selectedApp))

    #--------------------------------------------#
    # Select the model from the UI
    #--------------------------------------------#
    if selectedApp == 'Object Detection':
        st.info('This application performs object detection using advanced deep learning models. \
            It can detects more than 80 object from COCO dataset.')
        
        model = st.sidebar.selectbox(
            label='Select the model',
            options=['Caffe-MobileNetSSD', 'Darknet-YOLOv3-tiny'])

        st.sidebar.markdown("### :arrow_right: Parameters")
        confThresh = st.sidebar.slider(
            'Confidence', value=0.3, min_value=0.0, max_value=1.0)
        nmsThresh = st.sidebar.slider(
            'Non-maximum suppression', value=0.30, min_value=0.0, max_value=1.00, step=0.05)

        # Update the dictionnary
        guiParam.update(dict(confThresh=confThresh,
                                nmsThresh=nmsThresh,
                                model=model))

    #--------------------------------------------#

    elif selectedApp == 'Face Detection':
        st.info("This application performs face detection using advanced deep learning \
            models. It can detects face in the image")

        model = st.sidebar.selectbox(
                label='Select the model',
                options=['res10_300x300_ssd_iter_140000'])

        st.sidebar.markdown("### :arrow_right: Parameters")
        confThresh = st.sidebar.slider(
            'Confidence', value=0.80, min_value=0.0, max_value=1.00, step=0.05)
        
        # Update the dictionnary
        guiParam.update(dict(confThresh=confThresh,   model=model))
    
    #--------------------------------------------#

    elif selectedApp == 'Fire Detection':
        st.info(
            'This application performs fire detection using advanced deep learning models. ')
        model = st.sidebar.selectbox(
                    label='Select the model',
                    options=['Darknet-YOLOv3-tiny'])

        st.sidebar.markdown("### :arrow_right: Parameters")
        confThresh = st.sidebar.slider(
            'Confidence', value=0.5, min_value=0.0, max_value=1.0)
        nmsThresh = st.sidebar.slider(
            'Non-maximum suppression', value=0.30, min_value=0.0, max_value=1.00, step=0.05)
        
        # Update the dictionnary
        guiParam.update(dict(confThresh=confThresh,
                                nmsThresh=nmsThresh,
                                model=model))
    

    else:
        st.info('To start using the dashboard you must first \
                select an Application from the sidebar menu other than Empty')


    # Update the dictionnary
    guiParam.update(
        dict(selectedApp=selectedApp,
            dataSource=dataSource))

    return guiParam

# ------------------------------------------------------------------
# ------------------------------------------------------------------

def DataManager(guiParam):

    demo_images = {"Dog": "./data/dog.jpg",
                    "Family-picture": "./data/family.jpg",    
                    "Crosswalk": "./data/demo.jpg",
                    "Cat": "./data/cat.jpg",
                    "Car on fire": "./data/car_on_fire.jpg",
                    "Fire": "./data/fire.jpg",}

    url_demo_images = {
        "NY-City": "https://s4.thingpic.com/images/8a/Qcc4eLESvtjiGswmQRQ8ynCM.jpeg",
        "Paris-street": "https://www.discoverwalks.com/blog/wp-content/uploads/2018/08/best-streets-in-paris.jpg"}

    image = None

    #################################################################

    @st.cache(allow_output_mutation=True)
    def load_image_from_path(image_path):
        return cv.imread(image_path, cv.IMREAD_COLOR)

    @st.cache(allow_output_mutation=True)
    def load_image_from_upload(file):
        tmp = np.fromstring(file.read(), np.uint8)
        return cv.imdecode(tmp, 1)

    @st.cache(allow_output_mutation=True)
    def load_image_from_url(url_image):
        resp = urllib.request.urlopen(url_image)
        tmp = np.asarray(bytearray(resp.read()), dtype="uint8")
        return cv.imdecode(tmp, cv.IMREAD_COLOR)

    #################################################################
     
    #--------------------------------------------#
    # load image from database
    #--------------------------------------------#
    if guiParam["dataSource"] == 'Image: Demo':

        file_path = st.text_input('Enter the image PATH')

        if os.path.isfile(file_path):
            image = load_image_from_path(image_path=file_path)

        elif file_path == "":
            file_path_idx = st.selectbox(
                'Or select a demo image from the list', list(demo_images.keys()))
            file_path = demo_images[file_path_idx]

            image = load_image_from_path(image_path=file_path)
        else:
            raise ValueError("[Error] Please enter a valid image path")

    #--------------------------------------------#
    # upload image
    #--------------------------------------------#
    elif guiParam["dataSource"] == 'Image: Upload':

        file_path = st.file_uploader(
            'Upload an image', type=['png', 'jpg'])

        if file_path is not None:
            image = load_image_from_upload(file_path)
        elif file_path is None:
            raise ValueError(
                "[Error] Please upload a valid image ('png', 'jpg')")

    #--------------------------------------------#
    # Load an image from a URL
    #--------------------------------------------#

    elif guiParam["dataSource"] == 'Image: URL':
        file_path = st.text_input('Enter the image URL')

        if file_path != "":
            image = load_image_from_url(url_image=file_path)

        elif file_path == "":

            file_path_idx = st.selectbox(
                'Or select a URL from the list', list(url_demo_images.keys()))
            file_path = url_demo_images[file_path_idx]

            image = load_image_from_url(url_image=file_path)
        else:
            raise ValueError("[Error] Please enter a valid image URL")

    
    else:
        raise ValueError("Please select one source from the list")

    return image


# ------------------------------------------------------------------
# ------------------------------------------------------------------

import plugins as plugins

def AppManager(frame, guiParam):

    if guiParam['selectedApp'] == 'Object Detection':

        if guiParam['model'] == 'Caffe-MobileNetSSD':

            paramMobileNetSSD = dict(
                prototxt="models/MobileNetSSD_deploy.prototxt.txt",
                caffeModel="models/MobileNetSSD_deploy.caffemodel",
                confThresh=guiParam["confThresh"])

            objApp = plugins.Object_Detection_MobileNetSSD(
                paramMobileNetSSD)

        elif guiParam['model'] == 'Darknet-YOLOv3-tiny':
            paramYoloTiny = dict(labels='models/DarkNet/coco.names',
                                        modelCfg='models/DarkNet/yolov3-tiny.cfg',
                                        modelWeights="models/DarkNet/yolov3-tiny.weights",
                                        confThresh=guiParam['confThresh'],
                                        nmsThresh=guiParam['nmsThresh'])

            objApp = plugins.Object_Detection_YOLO(paramYoloTiny)

        else:
            raise ValueError(
                '[Error] Please selected one of the listed models')

    # -----------------------------------------------------

    elif guiParam['selectedApp'] == 'Face Detection':

        if guiParam['model'] == 'res10_300x300_ssd_iter_140000':

            param = dict(
                prototxt="models/deploy.prototxt.txt",
                caffeModel="models/res10_300x300_ssd_iter_140000.caffemodel",
                confThresh=guiParam["confThresh"])

            objApp = plugins.Face_Detection(param)
        else:
            raise ValueError(
                "[Error] Please selection one of the listed models")

    # -----------------------------------------------------

    elif guiParam['selectedApp'] == 'Fire Detection':
        @st.cache(allow_output_mutation=True)
        def getClasses(classesFile):
            """
            # Load names of classes
            """
            classes = None
            with open(classesFile, 'rt') as f:
                classes = f.read().rstrip('\n').split('\n')
            return classes

        labels = 'models/DarkNet/fire_detection/yolov3-tiny_obj.names'
        paramYoloTinyFire = dict(labels=labels,
                                        modelCfg='models/DarkNet/fire_detection/yolov3-tiny-obj.cfg',
                                        modelWeights="models/DarkNet/fire_detection/yolov3-tiny-obj_final.weights",
                                        confThresh=guiParam['confThresh'],
                                        nmsThresh=guiParam['nmsThresh'],
                                        colors=np.tile(colorBlue, (len(getClasses(labels)), 1)).tolist())

        objApp = plugins.Object_Detection_YOLO(paramYoloTinyFire)

        # -----------------------------------------------------

    else:
        raise Exception(
            '[Error] Please select one of the listed application')

    # Run the deep learning models
    bboxed_frame, output = objApp.run(frame, motion_state=True)

    return bboxed_frame, output

# ------------------------------------------------------------------
# ------------------------------------------------------------------


if __name__ == "__main__":
    
    # Configure the UI
    guiParam = UI()

    # Load the data (image)
    image = DataManager(guiParam)
    
    # Display the image
    img = st.empty()
    img.image(image, channels="BGR",  use_column_width=True)
    
    if guiParam['selectedApp'] != 'Empty':
    
        # Apply a computer vision model
        bboxed_image, output= AppManager(image, guiParam)

        # Diplay results
        img.image(bboxed_image, channels="BGR",  use_column_width=True)

        st.markdown('## :arrow_right: Processing info')
        st.dataframe(output['dataframe_plugin'])