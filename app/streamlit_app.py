# -*- coding: utf-8 -*-
import streamlit as st
import os
import cv2
from PIL import Image
from sys import path
import numpy as np
from streamlit_image_select import image_select
import pandas as pd

cpath = os.getcwd()
path.append(cpath + r'\codeX\utils')

from morph.face_landmark_detection import generate_face_correspondences
from morph.delaunay_triangulation import make_delaunay
from morph.face_morph_dev import generate_morph_sequence
import morph.analysis_morph
from src.codeX.utils.align_images_dev import align_images_dev

import os
from streamlit.components.v1 import html
import streamlit.components.v1 as components
from geopy.geocoders import ArcGIS
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import ColumnsAutoSizeMode

js_share = '''
        type="text/javascript"
        src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-63550914fb6a811c">
    '''

################################################################################################

def doMorphing(img1 , img2 , duration , frame_rate , output):
    [ size , img1 , img2 , points1 , points2 , list3 ] = generate_face_correspondences(img1 , img2)
    tri = make_delaunay(size[ 1 ] , size[ 0 ] , list3 , img1 , img2)
    res, res_origin = generate_morph_sequence(duration , frame_rate , img1 , img2 , points1 , points2 , tri , size , output)
    return res, res_origin


st.title('🎈 이뿌다 가상 성형 AI 🎈')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expended="true"]> div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expended="false"]> div:first-child{
        width: 350px
        margin-left = -350px
    }
    </style>
    """ ,
    unsafe_allow_html=True ,
)
st.sidebar.title('이뿌다 AI')
st.sidebar.subheader('가상성형/피부진단 AI')

count = 0


def mode_select(num):
    mode = [ '🖐️   About App' , '💉   가상 성형 AI' , '👧   피부 진단 AI', '🏥   입점 병원', '⏰   병원 상세 & 예약' ]
    app = st.sidebar.selectbox('choose the Appmode' ,
                                    mode ,
                                    key=str(num)
                                    )
    num += 1
    return app


app_mode = mode_select(count)

if app_mode == '🖐️   About App':
    st.markdown('"Experience the power of transformation with our interactive plastic surgery simulation service, allowing you to visualize the possibilities and make informed decisions about your desired changes."')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expended="true"]> div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expended="false"]> div:first-child{
            width: 350px
            margin-left = -350px
        }
        </style>
        """ ,
        unsafe_allow_html=True ,
    )
    st.video('https://youtu.be/1SGFEPEMaN4')

elif app_mode == '💉   가상 성형 AI':
    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expended="true"]> div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expended="false"]> div:first-child{
            width: 350px
            margin-left = -350px
        }
        </style>
        """ ,
        unsafe_allow_html=True ,
    )

    # st.markdown("**detected Faces**")
    # kpi1_text = st.markdown("0")

    # max_faces = st.sidebar.number_input('Maximum Number of Pictures' , value=2 , min_value=1 , key='num_input')
    # st.sidebar.markdown('---')
    # CHANGE_GRADE = st.sidebar.slider('얼마나 바꿔볼래?' , min_value=0.0 , max_value=0.99 , value=0.5 , key='slider')
    # st.sidebar.markdown('---')
    CHANGE_GRADE = .99
    src_image = st.sidebar.file_uploader("내 사진 올리기" , type=[ "jpg" , "jpeg" , "png" ] , key='myPhoto')
    src_target_image = st.sidebar.file_uploader("워너비 사진 올리기" , type=[ "jpg" , "jpeg" , "png" ] , key='celebPhoto')
    st.sidebar.markdown('---')
        
    with st.sidebar:
        html_string = '''
                <!-- Search Google -->
                <form method=get action="https://www.google.co.kr/imghp?hl=ko&tab=ri&authuser=0&ogbl" target="_blank" >
                <table bgcolor="#FFFFFF">
                    <tr>
                    <td width = "400">
                        <input type=text name=q size=25 maxlength=255 value="" /> <!-- 구글 검색 입력 창 -->
                        <input type=submit name=btnG value="연예인사진 Google 검색" /> <!-- 검색 버튼 -->
                    </td>
                    </tr>
                </table>
                </form>
                <!-- Search Google -->
            '''
        components.html(html_string)

    print(src_image)
    global morph_array
    col1, col2 = st.columns(2)        
    if src_image is not None:
        myImage = np.array(Image.open(src_image))
        st.sidebar.text('내 사진')
        st.sidebar.image(myImage)
        with col1:
            # st.text('내 사진')
            st.image(myImage,caption='내 사진',use_column_width=True)
    if src_target_image is not None:
        targetImage = np.array(Image.open(src_target_image))
        st.sidebar.text('워너비 사진')
        st.sidebar.image(targetImage)
        with col2:
            # placeholder_txt = st.text('워너비 사진')
            placeholder = st.image(targetImage,caption='워너비 사진',use_column_width=True)
            src_target_image = None
    else:            
            sampleImage = image_select(
                label="Select a Sample",
                images=[
                    np.array(Image.open(r"app/src/codeX/samples/1_01.png")),
                    np.array(Image.open(r"app/src/codeX/samples/2_01.png")),
                    np.array(Image.open(r"app/src/codeX/samples/3_01.png")),
                    np.array(Image.open(r"app/src/codeX/samples/4_01.png")),
                ],
                captions=["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
            )
            targetImage = sampleImage
            with col2:
                # placeholder_txt = st.text('워너비 사진')
                placeholder = st.image(sampleImage,caption='워너비 사진',use_column_width=True)
            
    if st.button("가상 성형 시작 :clap:" , key='morph_start'):
        empty_string = '''
        <div></div>
        '''
        components.html(empty_string)

        with st.spinner(':smiley: :smile: 이뿌게 성형 중이에용 ! :laughing: :grin:'):

            raw_image = align_images_dev(myImage, targetImage)
            MY_IMAGE = raw_image[0][0]
            if 'MY_IMAGE' not in st.session_state:
                st.session_state[ 'MY_IMAGE' ] = MY_IMAGE
            st.session_state[ 'MY_IMAGE' ] = MY_IMAGE
            TARGET_IMAGE = raw_image[1][0]
            if 'TARGET_IMAGE' not in st.session_state:
                st.session_state[ 'TARGET_IMAGE' ] = TARGET_IMAGE
            st.session_state[ 'TARGET_IMAGE' ] = TARGET_IMAGE

            image = np.array(MY_IMAGE.resize((500 , 500)))

            ########################################################################

            image1 = cv2.cvtColor(np.array(MY_IMAGE), cv2.COLOR_RGB2BGR)
            image2 = cv2.cvtColor(np.array(TARGET_IMAGE), cv2.COLOR_RGB2BGR)
            out_folder = cpath + r'\video_output.mp4'
            # doMorphing 변수 선언(100개의 numpy array)
            morph_array, morph_array_origin = doMorphing(image1 , image2 , int(5) , int(20) , out_folder)  ## Video Time
            if 'morph_array_origin' not in st.session_state:
                st.session_state[ 'morph_array_origin' ] = morph_array_origin
            st.session_state[ 'morph_array_origin' ] = morph_array_origin

            index = int(.5 * 100)
            # col1, co2 = st.columns(2)
            # with col1:
            st.image(morph_array_origin[index])
            # image_res = Image.fromarray(morph_array[index])
            # st.image(image_res)
            # morph_array_origin[0].save('frame.gif',
            #                    save_all=True, append_images=morph_array_origin[1:], optimize=False, duration=40, loop=1)
            # frame_image = Image.open('frame.gif')
            # with col2:
            #         st.image(frame_image)
        
        st.balloons()
    st.markdown('---')
    index = int(st.number_input('몇퍼센트 결과볼래?' , value=50 , step=1 , format="%d"))
    CHANGE_GRADE2 = st.slider('내사진 <<<<<----->>>>> 워너비' , min_value=0 , max_value=99 , value=index)
    if st.button("결과 보기!" , key='res'):
        # RAW_IMAGES_DIR = "db"  # args.raw_dir
        # ALIGNED_IMAGES_DIR = "db" + '/aligned_images'  # r'images\aligned_images' #args.aligned_dir
        # img_name = os.listdir(ALIGNED_IMAGES_DIR)

        # img1 = ALIGNED_IMAGES_DIR + '/' + img_name[ 0 ]
        # img2 = ALIGNED_IMAGES_DIR + '/' + img_name[ 1 ]
        # image1 = cv2.imread(img1)
        # image2 = cv2.imread(img2)
        # img_name_res = os.listdir('sequence_res\wo_line')
        index = CHANGE_GRADE2
        sequence_list = st.session_state[ 'morph_array_origin' ]

        print(str(index) + '.jpg')
        st.image(sequence_list[index])
        ana_image = cv2.cvtColor(np.array(sequence_list[index]), cv2.COLOR_RGB2BGR)
        res_tot = morph.analysis_morph.analysis(st.session_state[ 'MY_IMAGE' ] , ana_image)
        res_tot_2 = morph.analysis_morph.analysis(st.session_state[ 'MY_IMAGE' ] , st.session_state[ 'TARGET_IMAGE' ])

        left_eye_res = round(res_tot[ 0 ] , 2)
        if left_eye_res > 95:
            left_eye_res = 95
        if left_eye_res < 0:
            left_eye_res = 0
        R_eye_res = round(res_tot[ 1 ] , 2)
        if R_eye_res > 95:
            R_eye_res = 95
        if R_eye_res < 0:
            R_eye_res = 0        
        nose_bridge_res = round(res_tot[ 2 ] , 2)
        if nose_bridge_res > 95:
            nose_bridge_res = 95
        if nose_bridge_res < 0:
            nose_bridge_res = 0
        nos_res = round(res_tot[ 3 ] , 2)
        if nos_res > 95:
            nos_res = 95
        if nos_res < 0:
            nos_res = 0
        face_res = round(res_tot[ 4 ] , 2)
        if face_res > 95:
            face_res = 95
        if face_res < 0:
            face_res = 0
        L_eyebrow_res = round(res_tot[ 5 ] , 2)
        if L_eyebrow_res > 95:
            L_eyebrow_res = 95
        if L_eyebrow_res < 0:
            L_eyebrow_res = 0
        R_eyebrow_res = round(res_tot[ 6 ] , 2)
        if R_eyebrow_res > 95:
            R_eyebrow_res = 95
        if R_eyebrow_res < 0:
            R_eyebrow_res = 0
        U_mouth_res = round(res_tot[ 7 ] , 2)
        if U_mouth_res > 95:
            U_mouth_res = 95
        if U_mouth_res < 0:
            U_mouth_res = 0
        L_mouth_res = round(res_tot[ 8 ] , 2)
        if L_mouth_res > 95:
            L_mouth_res = 95
        if L_mouth_res < 0:
            L_mouth_res = 0

        col1 , col2 , col3 , col4 , col5 = st.columns(5)
        col1.metric("전체 유사도" , round(np.mean(res_tot) , 2) , round(np.mean(res_tot) - np.mean(res_tot_2)))
        col2.metric("왼쪽 눈 유사도" , left_eye_res , "-8%")
        col3.metric("오른쪽 눈 유사도" , R_eye_res , "4%")
        col4.metric("콧등 유사도" , nose_bridge_res)
        col5.metric("콧 망울 유사도" , nos_res , "-8%")

        col1 , col2 , col3 , col4 , col5 = st.columns(5)
        col1.metric("얼굴 아웃라인 유사도" , face_res , "4%")
        col2.metric("왼쪽 눈썹 유사도" , L_eyebrow_res , "-8%")
        col3.metric("오른쪽 눈썹 유사도" , R_eyebrow_res , "4%")
        col4.metric("윗 입술 유사도" , U_mouth_res)
        col5.metric("아랫 입술 유사도" , L_mouth_res , "-8%")
    my_html = '''
                <script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-63550914fb6a811c"></script>
                <div class="addthis_inline_share_toolbox_ww2q"></div>
                '''
    # Execute your app
    st.markdown('---')
    st.subheader("공유 해볼까~")
    html(my_html)
    st.markdown(my_html , unsafe_allow_html=True)  # JavaScript doesn't work

elif app_mode == '👧   피부 진단 AI':
    # import requests
    # url = "https://skin-analysis.p.rapidapi.com/face/effect/skin_analyze"
    MY_IMAGE = st.sidebar.file_uploader("내 사진 올리기" , type=[ "jpg" , "jpeg" , "png" ] , key='skinUploader')
    if MY_IMAGE is not None:
        # load image using PIL
        image_src = np.array(Image.open(MY_IMAGE))
        image_src_save = Image.open(MY_IMAGE)
        st.text('내 사진')
        st.image(image_src)
        RAW_IMAGES_DIR = "db"  # args.raw_dir
        ANALYSIS_IMAGES_DIR = "db" + '/analysis'  # r'images\aligned_images' #args.aligned_dir
        # Check whether the specified path exists or not
        isExist = os.path.exists(ANALYSIS_IMAGES_DIR)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(ANALYSIS_IMAGES_DIR)
            print("The new directory is created!")
        im1 = image_src_save.save(ANALYSIS_IMAGES_DIR + "/src.jpg")
        files = {"image": open(r'db\src.jpg' , 'rb')}
        # payload = {
        #     "max_face_num": "2",
        #     "face_field": "color,smooth,acnespotmole,wrinkle,eyesattr,blackheadpore,skinquality"
        # }
        # headers = {
        #     "X-RapidAPI-Key": "f675bb042bmsh5c733d40f9e9474p1cadd3jsnf5ce5040ef05",
        #     "X-RapidAPI-Host": "skin-analysis.p.rapidapi.com"
        # }

        # response = requests.post(url, data=payload, files=files, headers=headers)

        # print(response.json())

    import json

    x = {
        'request_id': '1682933969,83f2a4aa-bf95-42e5-bd53-f929c4ffc150' ,
        'result': {
            'skin_age': {
                'value': 20
            } ,
            'eye_pouch': {
                'value': 0 ,
                'confidence': 0.923059
            } ,
            'dark_circle': {
                'value': 2 ,
                'confidence': 1
            } ,
            'dark_circle_severity': {
                'value': 2 ,
                'confidence': 1
            } ,
            'forehead_wrinkle': {
                'value': 0 ,
                'confidence': 0.9821951
            } ,
            'crows_feet': {
                'value': 0 ,
                'confidence': 0.9954499
            } ,
            'eye_finelines': {
                'value': 0 ,
                'confidence': 0.48774555
            } ,
            'glabella_wrinkle': {
                'value': 0 ,
                'confidence': 0.99621344
            } ,
            'nasolabial_fold': {
                'value': 0 ,
                'confidence': 0.42546278
            } ,
            'skin_type': {
                'skin_type': 2 ,
                'details': [
                    {
                        'value': 0 ,
                        'confidence': 0.016151816
                    } , {
                        'value': 0 ,
                        'confidence': 0.0036073993
                    } , {
                        'value': 1 ,
                        'confidence': 0.97009605
                    } , {
                        'value': 0 ,
                        'confidence': 0.010144674
                    }
                ]
            } ,
            'pores_forehead': {
                'value': 1 ,
                'confidence': 1
            } ,
            'pores_left_cheek': {
                'value': 0 ,
                'confidence': 1
            } ,
            'pores_right_cheek': {
                'value': 0 ,
                'confidence': 1
            } ,
            'pores_jaw': {
                'value': 1 ,
                'confidence': 1
            } ,
            'blackhead': {
                'value': 0 ,
                'confidence': 1
            } ,
            'skintone_ita': {
                'ITA': 68.81677 ,
                'skintone': 0
            } ,
            'skin_hue_ha': {
                'HA': 43.042973 ,
                'skin_hue': 2
            } ,
            'acne': {
                'rectangle': [ ] ,
                'confidence': [ ] ,
                'polygon': [ ]
            } ,
            'mole': {
                'rectangle': [ ] ,
                'confidence': [ ] ,
                'polygon': [ ]
            } ,
            'brown_spot': {
                'rectangle': [
                    {
                        'left': 499 ,
                        'top': 576 ,
                        'width': 7 ,
                        'height': 8
                    }
                ] ,
                'confidence': [ 0.5159797 ] ,
                'polygon': [
                    [
                        {
                            'x': 504 ,
                            'y': 577
                        } , {
                        'x': 501 ,
                        'y': 582
                    } , {
                        'x': 500 ,
                        'y': 578
                    }
                    ]
                ]
            } ,
            'closed_comedones': {
                'rectangle': [ ] ,
                'confidence': [ ] ,
                'polygon': [ ]
            } ,
            'acne_mark': {
                'rectangle': [ ] ,
                'confidence': [ ] ,
                'polygon': [ ]
            } ,
            'acne_nodule': {
                'rectangle': [ ] ,
                'confidence': [ ] ,
                'polygon': [ ]
            } ,
            'acne_pustule': {
                'rectangle': [ ] ,
                'confidence': [ ] ,
                'polygon': [ ]
            } ,
            'blackhead_count': 40 ,
            'skintone': {
                'value': 1 ,
                'confidence': 0.99536175
            } ,
            'fine_line': {
                'forehead_count': 2 ,
                'left_undereye_count': 5 ,
                'right_undereye_count': 11 ,
                'left_cheek_count': 11 ,
                'right_cheek_count': 11 ,
                'left_crowsfeet_count': 11 ,
                'right_crowsfeet_count': 11 ,
                'glabella_count': 0
            } ,
            'wrinkle_count': {
                'forehead_count': 1 ,
                'left_undereye_count': 3 ,
                'right_undereye_count': 3 ,
                'left_mouth_count': 1 ,
                'right_mouth_count': 0 ,
                'left_nasolabial_count': 1 ,
                'right_nasolabial_count': 1 ,
                'glabella_count': 0 ,
                'left_cheek_count': 3 ,
                'right_cheek_count': 3 ,
                'left_crowsfeet_count': 1 ,
                'right_crowsfeet_count': 1
            } ,
            'oily_intensity': {
                't_zone': {
                    'area': 0.06 ,
                    'intensity': 0
                } ,
                'left_cheek': {
                    'area': 0 ,
                    'intensity': 0
                } ,
                'right_cheek': {
                    'area': 0.01 ,
                    'intensity': 0
                } ,
                'chin_area': {
                    'area': 0 ,
                    'intensity': 0
                }
            } ,
            'enlarged_pore_count': {
                'forehead_count': 185 ,
                'left_cheek_count': 36 ,
                'right_cheek_count': 26 ,
                'chin_count': 52
            } ,
            'right_dark_circle_rete': {
                'value': 0
            } ,
            'left_dark_circle_rete': {
                'value': 3
            } ,
            'right_dark_circle_pigment': {
                'value': 0
            } ,
            'left_dark_circle_pigment': {
                'value': 0
            } ,
            'right_dark_circle_structural': {
                'value': 0
            } ,
            'left_dark_circle_structural': {
                'value': 0
            } ,
            'dark_circle_mark': {
                'left_eye_rect': {
                    'left': 194 ,
                    'top': 376 ,
                    'width': 128 ,
                    'height': 111
                } ,
                'right_eye_rect': {
                    'left': 395 ,
                    'top': 377 ,
                    'width': 127 ,
                    'height': 111
                }
            } ,
            'water': {
                'water_severity': 2 ,
                'water_area': 0.017 ,
                'water_forehead': {
                    'area': 0.025
                } ,
                'water_leftcheek': {
                    'area': 0.006
                } ,
                'water_rightcheek': {
                    'area': 0.008
                }
            } ,
            'rough': {
                'rough_severity': 10 ,
                'rough_area': 0.12 ,
                'rough_forehead': {
                    'area': 0.106
                } ,
                'rough_leftcheek': {
                    'area': 0.176
                } ,
                'rough_rightcheek': {
                    'area': 0.058
                } ,
                'rough_jaw': {
                    'area': 0.209
                }
            } ,
            'left_mouth_wrinkle_severity': {
                'value': 3
            } ,
            'right_mouth_wrinkle_severity': {
                'value': 0
            } ,
            'forehead_wrinkle_severity': {
                'value': 0
            } ,
            'left_crows_feet_severity': {
                'value': 2
            } ,
            'right_crows_feet_severity': {
                'value': 2
            } ,
            'left_eye_finelines_severity': {
                'value': 0
            } ,
            'right_eye_finelines_severity': {
                'value': 0
            } ,
            'glabella_wrinkle_severity': {
                'value': 0
            } ,
            'left_nasolabial_fold_severity': {
                'value': 1
            } ,
            'right_nasolabial_fold_severity': {
                'value': 1
            } ,
            'left_cheek_wrinkle_severity': {
                'value': 0
            } ,
            'right_cheek_wrinkle_severity': {
                'value': 0
            } ,
            'left_crowsfeet_wrinkle_info': {
                'wrinkle_score': 27 ,
                'wrinkle_severity_level': 2 ,
                'wrinkle_norm_length': 1.119208160009301 ,
                'wrinkle_norm_depth': 0.35413469735720376 ,
                'wrinkle_pixel_density': 0.5917107262483214 ,
                'wrinkle_area_ratio': 0.025835866261398176 ,
                'wrinkle_deep_ratio': 0.782608695652174 ,
                'wrinkle_deep_num': 1 ,
                'wrinkle_shallow_num': 11
            } ,
            'right_crowsfeet_wrinkle_info': {
                'wrinkle_score': 50 ,
                'wrinkle_severity_level': 2 ,
                'wrinkle_norm_length': 2.13208057430361 ,
                'wrinkle_norm_depth': 0.28371888081075125 ,
                'wrinkle_pixel_density': 0.8910375563027902 ,
                'wrinkle_area_ratio': 0.03363767419509851 ,
                'wrinkle_deep_ratio': 0.5168539325842697 ,
                'wrinkle_deep_num': 1 ,
                'wrinkle_shallow_num': 11
            } ,
            'left_mouth_wrinkle_info': {
                'wrinkle_score': 100 ,
                'wrinkle_severity_level': 3 ,
                'wrinkle_norm_length': 0.68 ,
                'wrinkle_norm_depth': 0.7192853291829427 ,
                'wrinkle_pixel_density': 0.42257177563415504 ,
                'wrinkle_area_ratio': 0.4437299035369775 ,
                'wrinkle_deep_ratio': 1 ,
                'wrinkle_deep_num': 1 ,
                'wrinkle_shallow_num': 0
            } ,
            'left_nasolabial_wrinkle_info': {
                'wrinkle_score': 14 ,
                'wrinkle_severity_level': 1 ,
                'wrinkle_norm_length': 0.20620596075550524 ,
                'wrinkle_norm_depth': 0.13825809393524852 ,
                'wrinkle_pixel_density': 0.08360618275499082 ,
                'wrinkle_area_ratio': 0.06573426573426573 ,
                'wrinkle_deep_ratio': 1 ,
                'wrinkle_deep_num': 1 ,
                'wrinkle_shallow_num': 0
            } ,
            'right_nasolabial_wrinkle_info': {
                'wrinkle_score': 23 ,
                'wrinkle_severity_level': 1 ,
                'wrinkle_norm_length': 0.2110014482149356 ,
                'wrinkle_norm_depth': 0.2301247771836007 ,
                'wrinkle_pixel_density': 0.08154003532461557 ,
                'wrinkle_area_ratio': 0.012886025327704954 ,
                'wrinkle_deep_ratio': 1 ,
                'wrinkle_deep_num': 1 ,
                'wrinkle_shallow_num': 0
            } ,
            'score_info': {
                'dark_circle_score': 80 ,
                'skin_type_score': 30 ,
                'wrinkle_score': 75 ,
                'oily_intensity_score': 81 ,
                'pores_score': 85 ,
                'blackhead_score': 88 ,
                'acne_score': 100 ,
                'sensitivity_score': 95 ,
                'melanin_score': 80 ,
                'water_score': 98 ,
                'rough_score': 90 ,
                'total_score': 100
            } ,
            'left_eye_pouch_rect': {
                'left': 194 ,
                'top': 376 ,
                'width': 128 ,
                'height': 111
            } ,
            'right_eye_pouch_rect': {
                'left': 395 ,
                'top': 377 ,
                'width': 127 ,
                'height': 111
            } ,
            'melasma': {
                'value': 0 ,
                'confidence': 0.24145715
            } ,
            'freckle': {
                'value': 0 ,
                'confidence': 0.27422637
            } ,
            'image_quality': {
                'face_rect': {
                    'left': 151 ,
                    'top': 175 ,
                    'width': 422 ,
                    'height': 563
                } ,
                'face_ratio': 0.28690684 ,
                'hair_occlusion': 0.14233196 ,
                'face_orientation': {
                    'yaw': -1.5857886 ,
                    'pitch': 7.374387 ,
                    'roll': -0.617789
                }
            }
        } ,
        'face_rectangle': {
            'top': 336 ,
            'left': 158 ,
            'width': 409 ,
            'height': 408
        } ,
        'error_code': 0 ,
        'error_msg': ''
    }
    y = json.dumps(x)

    # parse x:
    y1 = json.loads(y)
    # from streamlit_apexjs import st_apexcharts
    # options = {
    #     "chart": {
    #         "toolbar": {
    #             "show": True
    #         }
    #     },

    #     "labels": [199]
    #     ,
    #     "legend": {
    #         "show": True,
    #         "position": "bottom",
    #     }
    # }

    # series = [80]

    # st_apexcharts(options, series, 'radialBar', '200', 'title')
    # st_apexcharts(options, series, 'radialBar', '200', 'title')
    # st_apexcharts(options, series, 'radialBar', '200', 'title')
    # st_apexcharts(options, series, 'radialBar', '200', 'title')

    # the result is a Python dictionary:
    print(y1[ 'result' ][ 'score_info' ])
    if st.button("진단 시작!" , key='Start2'):
        col1 , col2 , col3 , col4 , col5 , col6 = st.columns(6)
        col1.metric('다크 서클점수' , y1[ 'result' ][ 'score_info' ][ 'dark_circle_score' ])
        col2.metric('피부 퀄리티 점수' , y1[ 'result' ][ 'score_info' ][ 'skin_type_score' ])
        col3.metric('피부 퀄리티 점수' , y1[ 'result' ][ 'score_info' ][ 'skin_type_score' ])
        col4.metric('주름 점수' , y1[ 'result' ][ 'score_info' ][ 'wrinkle_score' ])
        col5.metric('지성피부 점수' , y1[ 'result' ][ 'score_info' ][ 'oily_intensity_score' ])
        col6.metric('모공 점수' , y1[ 'result' ][ 'score_info' ][ 'pores_score' ])

        col1 , col2 , col3 , col4 , col5 , col6 = st.columns(6)
        col1.metric('블랙헤드 점수' , y1[ 'result' ][ 'score_info' ][ 'blackhead_score' ])
        col2.metric('여드름 점수' , y1[ 'result' ][ 'score_info' ][ 'acne_score' ])
        col3.metric('피부 민감도 점수' , y1[ 'result' ][ 'score_info' ][ 'sensitivity_score' ])
        col4.metric('피부 색소침착 점수' , y1[ 'result' ][ 'score_info' ][ 'melanin_score' ])
        col5.metric('피부 수분 점수' , y1[ 'result' ][ 'score_info' ][ 'sensitivity_score' ])
        col6.metric('거친 피부 점수' , y1[ 'result' ][ 'score_info' ][ 'sensitivity_score' ])

        my_html = '''
                    <script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-63550914fb6a811c"></script>
                    <div class="addthis_inline_share_toolbox_ww2q"></div>
                    '''
        # Execute your app
        st.markdown('---')
        st.subheader("공유 해볼까~")
        html(my_html)
        st.markdown(my_html , unsafe_allow_html=True)  # JavaScript doesn't work

elif app_mode == '🏥   입점 병원':
    
    df_database = pd.read_csv("app/src/locations.csv",index_col=None)
    loc_database = df_database.loc[:,['lat','lon']]
    loc_database = pd.DataFrame(loc_database)
    loc_database = loc_database.dropna(axis=0)
    st.text('Our Branches')
    st.map(loc_database)
        
    # Configure grid options using GridOptionsBuilder
    builder = GridOptionsBuilder.from_dataframe(df_database)
    builder.configure_pagination(enabled=True)
    builder.configure_selection(selection_mode='single', use_checkbox=False)
    builder.configure_column('System Name', editable=False)
    builder.configure_grid_options(rowHeight=30)
    builder.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)

    grid_options = builder.build()
    column_defs = grid_options["columnDefs"]
    columns_to_hide = ["Index","No.","System Name","lat","lon"]
    # update the column definitions to hide the specified columns
    for col in column_defs:
        if col["headerName"] in columns_to_hide:
            col["hide"] = True
    selected_rows = AgGrid(df_database, gridOptions=grid_options,
                          columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                        )
    if selected_rows['selected_rows']:
        num_selected = selected_rows['selected_rows'][0]['Index']
        # print(num_selected,loc_database.loc[[num_selected],:])
        st.text('Hospital Location 🌏')
        st.map(loc_database.loc[[num_selected],:]) 
elif app_mode == '⏰   병원 상세 & 예약':
#         st.markdown(
#     """
#         <html lang="en">
#         <head>
         
#             <meta charset="utf-8">
#             <meta name="viewport" content="width=device-width, initial-scale=1">
          
#             <title>Ippuda AI</title>
          
#             <link rel="stylesheet" media="screen" href="https://cpwebassets.codepen.io/assets/fullpage/fullpage-9a6f5ffcc359b81b95df109e4b8ce436f472f4f828fcc75227cd5266125d459b.css">
            
          
#               <link rel="apple-touch-icon" type="image/png" href="https://cpwebassets.codepen.io/assets/favicon/apple-touch-icon-5ae1a0698dcc2402e9712f7d01ed509a57814f994c660df9f7a952f3060705ee.png">
          
#               <meta name="apple-mobile-web-app-title" content="CodePen">
          
#               <link rel="shortcut icon" type="image/x-icon" href="https://cpwebassets.codepen.io/assets/favicon/favicon-aec34940fbc1a6e787974dcd360f2c6b63348d4b1f4e06c77743096d55480f33.ico">
          
#               <link rel="mask-icon" type="image/x-icon" href="https://cpwebassets.codepen.io/assets/favicon/logo-pin-b4b4269c16397ad2f0f7a01bcdf513a1994f4c94b8af2f191c09eb0d601762b1.svg" color="#111">
        
#             <style>
#               html { font-size: 15px; }
#               html, body { margin: 0; padding: 0; min-height: 100%; }
#               body { height:100%; display: flex; flex-direction: column; }
#               .referer-warning {
#                 background: black;
#                 box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
#                 padding: 0.75em;
#                 color: white;
#                 text-align: center;
#                 font-family: 'Lato', 'Lucida Grande', 'Lucida Sans Unicode', Tahoma, system-ui, Sans-Serif;
#                 line-height: 1.2;
#                 font-size: 1rem;
#                 position: relative;
#                 z-index: 2;
#               }
#               .referer-warning span { font-family: initial; }
#               .referer-warning h1 { font-size: 1.2rem; margin: 0; }
#               .referer-warning a { color: #56bcf9; }
#             </style>
#           <script src="chrome-extension://ecmeogcbcoalojmkfkmancobmiahaigg/content_script/content_script.js"></script></head>
        
#             <div id="result-iframe-wrap" role="main">
#               <iframe id="result" srcdoc="<!DOCTYPE html>
#           <html lang=&quot;en&quot; >
          
#           <head>
#             <meta charset=&quot;UTF-8&quot;>
            
          
#               <link rel=&quot;apple-touch-icon&quot; type=&quot;image/png&quot; href=&quot;https://cpwebassets.codepen.io/assets/favicon/apple-touch-icon-5ae1a0698dcc2402e9712f7d01ed509a57814f994c660df9f7a952f3060705ee.png&quot; />
          
#               <meta name=&quot;apple-mobile-web-app-title&quot; content=&quot;CodePen&quot;>
          
#               <link rel=&quot;shortcut icon&quot; type=&quot;image/x-icon&quot; href=&quot;https://cpwebassets.codepen.io/assets/favicon/favicon-aec34940fbc1a6e787974dcd360f2c6b63348d4b1f4e06c77743096d55480f33.ico&quot; />
          
#               <link rel=&quot;mask-icon&quot; type=&quot;image/x-icon&quot; href=&quot;https://cpwebassets.codepen.io/assets/favicon/logo-pin-b4b4269c16397ad2f0f7a01bcdf513a1994f4c94b8af2f191c09eb0d601762b1.svg&quot; color=&quot;#111&quot; />
          
          
          
            
#             <title>CodePen - Dental Care Layout</title>
#               <link rel=&quot;canonical&quot; href=&quot;https://codepen.io/max131/pen/NWpmvML&quot; />
            
#             <link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/normalize/5.0.0/normalize.min.css&quot;>
          
#             <link rel='stylesheet' href='https://assets.calendly.com/assets/external/widget.css'>
            
#           <style>
#           :root {
#             --color-primary: hsl(225, 74%, 57%);
#             --color-primary-d: hsl(225, 74%, 37%);
#             --color-primary-l: hsl(225, 74%, 77%);
#             --color-title: hsl(0, 0%, 25%);
#             --color-text: hsl(0, 0%, 35%);
#             --color-text-l: hsl(0, 0%, 65%);
#             --color-bg: hsl(355, 71%, 99%);
#             --color-container: hsl(0, 0%, 100%);
#             --shadow: hsla(0, 0%, 20%, 0.3);
#             --shadow-l: hsla(0, 0%, 20%, 0.1);
#             --shadow-d: hsla(0, 0%, 20%, 0.6);
#             --font-primary: &quot;Poppins&quot;;
#             --border-r-s: 0.25rem;
#             --border-r-m: 0.5rem;
#             --border-r-l: 1rem;
#             --border-r-f: 99rem;
#           }
          
#           * {
#             box-sizing: border-box;
#           }
          
#           html {
#             font: 18px/1.6 &quot;Poppins&quot;, helvetica, system-ui, sans-serif;
#             scroll-behavior: smooth;
#           }
          
#           body {
#             min-height: 100vh;
#             /*background: var(--color-bg) url(&quot;data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%231788FA' fill-opacity='1' d='M0,192L60,186.7C120,181,240,171,360,186.7C480,203,600,245,720,240C840,235,960,181,1080,149.3C1200,117,1320,107,1380,101.3L1440,96L1440,320L1380,320C1320,320,1200,320,1080,320C960,320,840,320,720,320C600,320,480,320,360,320C240,320,120,320,60,320L0,320Z'%3E%3C/path%3E%3C/svg%3E&quot;) no-repeat bottom/100%;*/
#             color: var(--color-text);
#             margin: 0;
#           }
          
#           body.dark {
#             background: var(--color-title);
#             color: var(--color-bg);
#           }
          
#           body.dark .navbar {
#             background: var(--color-title);
#           }
          
#           body.dark .navbar a {
#             color: var(--color-bg);
#           }
          
#           body.dark .menu .menu__link {
#             color: initial;
#           }
          
#           body.dark .title {
#             color: var(--color-primary);
#           }
          
#           body.dark footer {
#             color: var(--color-primary-l);
#           }
          
#           a {
#             text-decoration: none;
#             color: var(--color-primary);
#           }
          
#           a:active:not(.button) {
#             color: var(--color-primary-d);
#           }
          
#           p {
#             margin-bottom: 1.5rem;
#           }
          
#           figure {
#             margin: 0;
#           }
          
#           img {
#             display: inline-block;
#             max-width: 100%;
#             border-radius: var(--border-r-m);
#             box-shadow: 0 4px 20px 1px var(--shadow-d);
#           }
          
#           img.circle {
#             border-radius: 50%;
#             box-shadow: 0 2px 15px 2px var(--shadow);
#           }
          
#           .center {
#             text-align: center;
#           }
          
#           .icon,
#           .service-icon {
#             width: 1rem;
#             height: 1rem;
#             display: inline-flex;
#             place-items: center;
#             margin: 0.15rem;
#           }
          
#           .title {
#             color: var(--color-primary-d);
#           }
          
#           .subtitle {
#             font-size: large;
#           }
          
#           .button {
#             display: inline-block;
#             padding: 0.75rem 1.5rem;
#             border-radius: var(--border-r-f);
#             background: var(--color-primary);
#             color: var(--color-bg);
#             text-decoration: none;
#           }
          
#           .button:hover {
#             background: var(--color-primary-d);
#             box-shadow: 0 2px 10px var(--shadow-d);
#           }
          
#           .button:active {
#             transform: translateY(3px);
#           }
          
#           /********/
#           /*Navbar*/
#           /********/
#           .navbar {
#             position: sticky;
#             top: 0;
#             padding: 0.5rem;
#             background: var(--color-bg);
#             box-shadow: 0 2px 15px -5px var(--shadow-l);
#             z-index: 10;
#           }
          
#           .navbar__container {
#             max-width: 54rem;
#             margin: 0 auto;
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#           }
          
#           .brand-title {
#             font-size: 1.25rem;
#             font-weight: bold;
#             text-decoration: none;
#             color: var(--color-title);
#           }
          
#           #menubtn {
#             display: none;
#           }
          
#           .menulbl {
#             cursor: pointer;
#             background: url(&quot;data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 448 512' width='100' title='bars'%3E%3Cpath d='M16 132h416c8.837 0 16-7.163 16-16V76c0-8.837-7.163-16-16-16H16C7.163 60 0 67.163 0 76v40c0 8.837 7.163 16 16 16zm0 160h416c8.837 0 16-7.163 16-16v-40c0-8.837-7.163-16-16-16H16c-8.837 0-16 7.163-16 16v40c0 8.837 7.163 16 16 16zm0 160h416c8.837 0 16-7.163 16-16v-40c0-8.837-7.163-16-16-16H16c-8.837 0-16 7.163-16 16v40c0 8.837 7.163 16 16 16z' /%3E%3C/svg%3E&quot;) no-repeat center/cover;
#           }
          
#           #menubtn:checked + .menulbl {
#             background-image: url(&quot;data:image/svg+xml,%3Csvg aria-hidden='true' focusable='false' data-prefix='fas' data-icon='times' class='svg-inline--fa fa-times fa-w-11' role='img' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 352 512'%3E%3Cpath fill='currentColor' d='M242.72 256l100.07-100.07c12.28-12.28 12.28-32.19 0-44.48l-22.24-22.24c-12.28-12.28-32.19-12.28-44.48 0L176 189.28 75.93 89.21c-12.28-12.28-32.19-12.28-44.48 0L9.21 111.45c-12.28 12.28-12.28 32.19 0 44.48L109.28 256 9.21 356.07c-12.28 12.28-12.28 32.19 0 44.48l22.24 22.24c12.28 12.28 32.2 12.28 44.48 0L176 322.72l100.07 100.07c12.28 12.28 32.2 12.28 44.48 0l22.24-22.24c12.28-12.28 12.28-32.19 0-44.48L242.72 256z'%3E%3C/path%3E%3C/svg%3E&quot;);
#           }
          
#           #menubtn:checked ~ .menu {
#             /*clip-path: circle(100% at center);*/
#             top: 105%;
#           }
          
#           .menu {
#             list-style: none;
#             margin: 1rem;
#             padding: 1rem;
#             width: calc(100% - 2rem);
#             border-radius: var(--border-r-s);
#             background: var(--color-container);
#             box-shadow: 0 0 20px -8px var(--shadow-d);
#             position: absolute;
#             top: -500%;
#             left: 0;
#             /*clip-path: circle(0 at center);*/
#             transition: top 0.5s cubic-bezier(0.76, -0.51, 0.26, 1.78);
#           }
          
#           .menu__item {
#             text-align: center;
#           }
          
#           .menu__link {
#             display: block;
#             padding: 0.25rem;
#             margin: 0.25rem;
#             color: var(--color-title);
#           }
          
#           .menu__link:hover {
#             color: var(--color-primary-d);
#           }
          
#           .section {
#             padding: 1rem;
#             display: grid;
#             margin: 2rem auto;
#             max-width: 54rem;
#           }
          
#           /*header*/
#           .header {
#             gap: 2rem;
#             grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
#           }
          
#           .header__col:last-child {
#             margin-top: 2rem;
#             display: grid;
#             place-items: center;
#           }
          
#           .image-blob {
#             border-radius: 50% 50% 50% 50%/70% 70% 30% 30%;
#             height: calc(100vw - 2rem);
#             width: calc(100vw - 2rem);
#             max-height: 18rem;
#             max-width: 18rem;
#             margin: 0 auto;
#             object-fit: cover;
#             object-position: top;
#           }
          
#           /**********/
#           /*Services*/
#           /**********/
#           .services {
#             gap: 1rem;
#             grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
#             justify-content: center;
#           }
          
#           .service {
#             text-align: center;
#           }
          
#           .service-icon {
#             width: 4rem;
#             height: 4rem;
#             color: var(--color-text-d);
#           }
          
#           .service:nth-child(1) .service-icon {
#             color: lightpink;
#           }
          
#           .service:nth-child(2) .service-icon {
#             color: lightsalmon;
#           }
          
#           .service:nth-child(3) .service-icon {
#             color: lightblue;
#           }
          
#           .service .subtitle {
#             color: var(--color-text-d);
#           }
          
#           /******/
#           /*info*/
#           /******/
#           .info {
#             gap: 5rem;
#             place-items: center;
#           }
          
#           .info:nth-of-type(2) img {
#             border-radius: 65% 35% 35% 65%/37% 68% 32% 63%;
#           }
          
#           .info:nth-of-type(3) img {
#             border-radius: 44% 56% 59% 41%/61% 33% 67% 39%;
#           }
          
#           /**************/
#           /*Testimonials*/
#           /**************/
#           .testimonials {
#             overflow: hidden;
#             padding: 0;
#             margin: 0 auto 2rem;
#             max-width: 35rem;
#           }
          
#           .testimonials__container {
#             display: flex;
#             animation: carousel 15s infinite;
#           }
          
#           .testimonial {
#             flex: 1 0 100%;
#             padding: 2rem 0 1rem;
#             text-align: center;
#             background: linear-gradient(var(--color-container), var(--shadow-l));
#             border-radius: var(--border-r-s);
#             box-shadow: 0 0 15px -3px var(--shadow-l);
#           }
          
#           @keyframes carousel {
#             1% {
#               transform: translateX(0%);
#             }
#             31% {
#               transform: translateX(0%);
#             }
#             35% {
#               transform: translateX(-100%);
#             }
#             64% {
#               transform: translateX(-100%);
#             }
#             68% {
#               transform: translateX(-200%);
#             }
#             97% {
#               transform: translateX(-200%);
#             }
#           }
#           /********/
#           /*Footer*/
#           /********/
#           .footer {
#             max-width: 100%;
#             margin-top: 5rem;
#             margin-bottom: 0;
#             grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
#             background: var(--color-primary);
#             color: var(--color-bg);
#             box-shadow: 0 -4px 15px var(--shadow);
#           }
          
#           .footer ul {
#             list-style: none;
#             padding: 0;
#           }
          
#           .address p {
#             margin: 0;
#           }
          
#           .social-link {
#             display: flex;
#             align-items: center;
#             color: inherit;
#           }
          
#           .social-link .icon {
#             margin-right: 0.5rem;
#           }
          
#           /***************/
#           /*Media Queries*/
#           /***************/
#           @media screen and (min-width: 768px) {
#             .header__col:last-child {
#               margin-top: 0;
#             }
          
#             .info {
#               grid-template-columns: repeat(3, minmax(180px, 1fr));
#             }
          
#             .info-element {
#               grid-column: span 2;
#             }
          
#             .info:nth-of-type(3) figure {
#               order: -1;
#             }
          
#             .footer {
#               place-items: center;
#             }
#           }
#           @media screen and (min-width: 960px) {
#             .navbar {
#               margin-bottom: 3rem;
#             }
          
#             .menulbl {
#               display: none;
#             }
          
#             .menu {
#               position: initial;
#               display: flex;
#               width: initial;
#               margin: initial;
#               padding: initial;
#               background: none;
#               box-shadow: none;
#             }
#           }
#           </style>
          
#             <script>
#             window.console = window.console || function(t) {};
#           </script>
          
            
            
#           </head>
          
#           <body translate=&quot;no&quot;>
#             <nav class=&quot;navbar&quot;>
#                   <div class=&quot;navbar__container&quot;>
#                       <a class=&quot;brand-title&quot; href=&quot;#!&quot;>Ippuda AI</a>
#                           <input type=&quot;checkbox&quot; id=&quot;menubtn&quot;>
#                           <label for=&quot;menubtn&quot; class=&quot;icon menulbl&quot;>
#                           </label>
#                       <ul class=&quot;menu&quot;>
#                           <li class=&quot;menu__item&quot;>
#                               <a class=&quot;menu__link&quot; href=&quot;#!&quot;>수술 범위</a>
#                           </li>
#                           <li class=&quot;menu__item&quot;>
#                               <a class=&quot;menu__link&quot; href=&quot;#!&quot;>상담 예약</a>
#                           </li>
#                           <li class=&quot;menu__item&quot;>
#                               <a class=&quot;menu__link&quot; href=&quot;#appointments&quot;>자세히</a>
#                           </li>
#                       </ul>
#                   </div>
#               </nav>
#           <header class=&quot;section header&quot;>
#               <div class=&quot;header__col&quot;>
#                   <h1 class=&quot;title&quot;>당신의 환한 미소를 위해</h1>
#                   <p>저희 성형외과는 최신 기술과 전문성을 바탕으로 고품질의 성형외과 서비스를 제공하는 병원입니다. 개인 맞춤형 상담과 안전한 수술을 통해 환자의 아름다움과 자신감 향상을 돕고 있습니다. 친절한 의료진과 첨단 장비를 보유하고 있어 최상의 결과를 위해 노력하고 있습니다. 환자 중심의 서비스와 만족도에 최우선으로 신경쓰며, 안전하고 만족스러운 성형외과 경험을 제공합니다.</p>
#                   <p>
#                       <a href=&quot;#!&quot; class=&quot;button&quot; onclick=&quot;Calendly.initPopupWidget({url: 'https://calendly.com/hnovation/30min?back=1&month=2023-07?hide_landing_page_details=1'});return false;&quot;>상담예약</a>
#                   </p>
#               </div>
#               <div class=&quot;header__col&quot;>
#                   <figure class=&quot;header-img&quot;>
#                       <img src=&quot;https://images.unsplash.com/photo-1494790108377-be9c29b29330?crop=entropy&amp;cs=srgb&amp;fm=jpg&amp;ixid=MnwxNDU4OXwwfDF8cmFuZG9tfHx8fHx8fHx8MTYyNDMwODczOA&amp;ixlib=rb-1.2.1&amp;q=85&quot; alt=&quot;&quot; class=&quot;image image-blob&quot;>
#                   </figure>
#               </div>		
#           </header>
#           <section class=&quot;section services&quot;>
#               <div class=&quot;service&quot;>
#                   <figure>
#                       <div class=&quot;service-icon&quot;>
#                           <svg aria-hidden=&quot;true&quot; focusable=&quot;false&quot; data-prefix=&quot;fas&quot; data-icon=&quot;stethoscope&quot; class=&quot;svg-inline--fa fa-stethoscope fa-w-16&quot; role=&quot;img&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot; viewBox=&quot;0 0 512 512&quot;><path fill=&quot;currentColor&quot; d=&quot;M447.1 112c-34.2.5-62.3 28.4-63 62.6-.5 24.3 12.5 45.6 32 56.8V344c0 57.3-50.2 104-112 104-60 0-109.2-44.1-111.9-99.2C265 333.8 320 269.2 320 192V36.6c0-11.4-8.1-21.3-19.3-23.5L237.8.5c-13-2.6-25.6 5.8-28.2 18.8L206.4 35c-2.6 13 5.8 25.6 18.8 28.2l30.7 6.1v121.4c0 52.9-42.2 96.7-95.1 97.2-53.4.5-96.9-42.7-96.9-96V69.4l30.7-6.1c13-2.6 21.4-15.2 18.8-28.2l-3.1-15.7C107.7 6.4 95.1-2 82.1.6L19.3 13C8.1 15.3 0 25.1 0 36.6V192c0 77.3 55.1 142 128.1 156.8C130.7 439.2 208.6 512 304 512c97 0 176-75.4 176-168V231.4c19.1-11.1 32-31.7 32-55.4 0-35.7-29.2-64.5-64.9-64zm.9 80c-8.8 0-16-7.2-16-16s7.2-16 16-16 16 7.2 16 16-7.2 16-16 16z&quot;></path></svg>
#                       </div>
#                   </figure>
#                   <h2 class=&quot;subtitle&quot;>성형 전문의</h2>
#                   <p>환자들의 아름다움과 자신감 향상을 돕기 위해 최선을 다하고 있습니다. 다년간의 경험과 지식을 바탕으로 개인 맞춤형 상담과 안전한 수술을 제공합니다. 환자의 요구와 목표를 이해하고, 최신 기술과 장비를 활용하여 최상의 결과를 창출합니다.</p>
#               </div>
#               <div class=&quot;service&quot;>
#                   <figure>
#                       <div class=&quot;service-icon&quot;>
#                           <svg aria-hidden=&quot;true&quot; focusable=&quot;false&quot; data-prefix=&quot;fas&quot; data-icon=&quot;notes-medical&quot; class=&quot;svg-inline--fa fa-notes-medical fa-w-12&quot; role=&quot;img&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot; viewBox=&quot;0 0 384 512&quot;><path fill=&quot;currentColor&quot; d=&quot;M336 64h-80c0-35.3-28.7-64-64-64s-64 28.7-64 64H48C21.5 64 0 85.5 0 112v352c0 26.5 21.5 48 48 48h288c26.5 0 48-21.5 48-48V112c0-26.5-21.5-48-48-48zM192 40c13.3 0 24 10.7 24 24s-10.7 24-24 24-24-10.7-24-24 10.7-24 24-24zm96 304c0 4.4-3.6 8-8 8h-56v56c0 4.4-3.6 8-8 8h-48c-4.4 0-8-3.6-8-8v-56h-56c-4.4 0-8-3.6-8-8v-48c0-4.4 3.6-8 8-8h56v-56c0-4.4 3.6-8 8-8h48c4.4 0 8 3.6 8 8v56h56c4.4 0 8 3.6 8 8v48zm0-192c0 4.4-3.6 8-8 8H104c-4.4 0-8-3.6-8-8v-16c0-4.4 3.6-8 8-8h176c4.4 0 8 3.6 8 8v16z&quot;></path></svg>
#                       </div>
#                   </figure>
#                   <h2 class=&quot;subtitle&quot;>환자 우선</h2>
#                   <p>환자들의 아름다움과 자신감 향상을 돕기 위해 최선을 다하고 있습니다. 다년간의 경험과 지식을 바탕으로 개인 맞춤형 상담과 안전한 수술을 제공합니다. 환자의 요구와 목표를 이해하고, 최신 기술과 장비를 활용하여 최상의 결과를 창출합니다.</p>
#               </div>
#               <div class=&quot;service&quot;>
#                   <figure>
#                       <div class=&quot;service-icon&quot;>
#                           <svg aria-hidden=&quot;true&quot; focusable=&quot;false&quot; data-prefix=&quot;fas&quot; data-icon=&quot;teeth&quot; class=&quot;svg-inline--fa fa-teeth fa-w-20&quot; role=&quot;img&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot; viewBox=&quot;0 0 640 512&quot;><path fill=&quot;currentColor&quot; d=&quot;M544 0H96C42.98 0 0 42.98 0 96v320c0 53.02 42.98 96 96 96h448c53.02 0 96-42.98 96-96V96c0-53.02-42.98-96-96-96zM160 368c0 26.51-21.49 48-48 48s-48-21.49-48-48v-64c0-8.84 7.16-16 16-16h64c8.84 0 16 7.16 16 16v64zm0-128c0 8.84-7.16 16-16 16H80c-8.84 0-16-7.16-16-16v-64c0-26.51 21.49-48 48-48s48 21.49 48 48v64zm144 120c0 30.93-25.07 56-56 56s-56-25.07-56-56v-56c0-8.84 7.16-16 16-16h80c8.84 0 16 7.16 16 16v56zm0-120c0 8.84-7.16 16-16 16h-80c-8.84 0-16-7.16-16-16v-88c0-30.93 25.07-56 56-56s56 25.07 56 56v88zm144 120c0 30.93-25.07 56-56 56s-56-25.07-56-56v-56c0-8.84 7.16-16 16-16h80c8.84 0 16 7.16 16 16v56zm0-120c0 8.84-7.16 16-16 16h-80c-8.84 0-16-7.16-16-16v-88c0-30.93 25.07-56 56-56s56 25.07 56 56v88zm128 128c0 26.51-21.49 48-48 48s-48-21.49-48-48v-64c0-8.84 7.16-16 16-16h64c8.84 0 16 7.16 16 16v64zm0-128c0 8.84-7.16 16-16 16h-64c-8.84 0-16-7.16-16-16v-64c0-26.51 21.49-48 48-48s48 21.49 48 48v64z&quot;></path></svg>
#                       </div>
#                   </figure>
#                   <h2 class=&quot;subtitle&quot;>최고의 만족</h2>
#                   <p>수술 후의 관리와 결과에 대한 세심한 배려와 지원으로 환자들은 우리 성형외과를 높게 평가하고 있습니다. 우리와 함께하면 안전하고 만족스러운 성형외과 경험을 할 수 있습니다.</p>
#               </div>
#           </section>
#           <section class=&quot;section info&quot;>
#               <div class=&quot;info-element&quot;>
#                   <h2>성형 전문</h2>
#                   <p>저희 성형외과는 최고의 전문의 팀으로 구성되어 있습니다. 각 전문의는 다년간의 경험과 탁월한 전문성을 갖추고 있으며, 최신 동향과 기술에 대한 꾸준한 업데이트를 실시합니다. 환자 중심의 접근 방식과 정확한 진단을 통해 개인 맞춤형 치료를 제공하며, 환자의 안전과 만족도를 최우선으로 생각합니다.</p>
#               </div>
#               <figure>
#                   <img src=&quot;https://images.unsplash.com/photo-1606265752439-1f18756aa5fc?crop=entropy&amp;cs=srgb&amp;fm=jpg&amp;ixid=MnwxNDU4OXwwfDF8cmFuZG9tfHx8fHx8fHx8MTYyNDQyMTUxNA&amp;ixlib=rb-1.2.1&amp;q=85&quot; alt=&quot;&quot; class=&quot;image&quot;>
#               </figure>
#           </section>
#           <h2 class=&quot;section center&quot;>수술 후기</h2>
#           <section class=&quot;section testimonials&quot;>
#               <div class=&quot;testimonials__container&quot;>
                  
#                       <div class=&quot;testimonial&quot;>
#                           <figure>
#                               <img src=&quot;https://picsum.photos/id/453/100&quot; alt=&quot;&quot; class=&quot;circle&quot;>
#                           </figure>
#                           <h3>김지영</h3>
#                           <p>병원 C의 성형외과는 전문성과 고객 서비스에 대해 아주 좋은 평가를 받을 만큼 좋은 병원입니다. 의사와 간호사들이 매우 친절하고 지식이 풍부했으며, 수술 전과 후에도 저를 위해 시간을 내어 상담을 해 주었습니다. 수술 결과도 제 예상을 뛰어넘어서 매우 만족스러웠습니다.</p>
#                           <p>⭐ ⭐ ⭐ ⭐ ⭐</p>
#                       </div>
          
#                       <div class=&quot;testimonial&quot;>
#                           <figure>
#                               <img src=&quot;https://picsum.photos/id/1027/100&quot; alt=&quot;&quot; class=&quot;circle&quot;>
#                           </figure>
#                           <h3>박수민</h3>
#                           <p>병원 B의 성형외과는 전문성과 전문성에 빛을 발하고 있습니다. 수술 전에 상세한 상담을 진행해 주었고, 어떤 절차를 거칠지에 대해 명확하게 설명해 주었습니다. 수술 시에도 신경을 많이 써 주었고, 수술 후에는 늘 제 곁에 있어 주었습니다. 최종 결과도 매우 만족스러웠고, 절대로 후회하지 않았습니다.</p>
#                           <p>⭐ ⭐ ⭐ ⭐ ⭐</p>
#                       </div>
          
#                       <div class=&quot;testimonial&quot;>
#                           <figure>
#                               <img src=&quot;https://picsum.photos/id/883/100&quot; alt=&quot;&quot; class=&quot;circle&quot;>
#                           </figure>
#                           <h3>이준호</h3>
#                           <p>병원 A의 성형외과는 정말로 훌륭한 서비스를 제공해 주었습니다. 의사와 간호사들은 매우 전문적이었고, 내가 원하는 결과를 얻기 위해 정확한 지침을 제공해 주었습니다. 수술 후에도 궁금한 점이나 우려사항이 있을 때마다 친절하게 답변해 주었고, 전반적으로 매우 만족스러운 경험이었습니다.</p>
#                           <p>⭐ ⭐ ⭐ ⭐ ⭐</p>
#                       </div>
          
#               </div>
#           </section>
#           <section class=&quot;section&quot;>
#               <h2 class=&quot;center&quot; id=&quot;appointments&quot;>상담 예약</h2>
#               <div class=&quot;calendly-inline-widget&quot; data-url=&quot;https://calendly.com/hnovation/30min?back=1&month=2023-07&quot; style=&quot;min-width:320px;height:630px;&quot;></div>	
#           </section>
#           <footer class=&quot;section footer&quot;>
#               <address class=&quot;address&quot;>
#                   <p>Ippuda AI</p>
#                   <p>U&U</p>
#                   <p>C.P. 12345</p>
#               </address>
#               <ul>
#                   <li>
#                       <a href=&quot;#!&quot; class=&quot;social-link&quot;>
#                           <span class=&quot;icon&quot;>
#                               <svg aria-hidden=&quot;true&quot; focusable=&quot;false&quot; data-prefix=&quot;fas&quot; data-icon=&quot;phone-square-alt&quot; class=&quot;svg-inline--fa fa-phone-square-alt fa-w-14&quot; role=&quot;img&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot; viewBox=&quot;0 0 448 512&quot;><path fill=&quot;currentColor&quot; d=&quot;M400 32H48A48 48 0 0 0 0 80v352a48 48 0 0 0 48 48h352a48 48 0 0 0 48-48V80a48 48 0 0 0-48-48zm-16.39 307.37l-15 65A15 15 0 0 1 354 416C194 416 64 286.29 64 126a15.7 15.7 0 0 1 11.63-14.61l65-15A18.23 18.23 0 0 1 144 96a16.27 16.27 0 0 1 13.79 9.09l30 70A17.9 17.9 0 0 1 189 181a17 17 0 0 1-5.5 11.61l-37.89 31a231.91 231.91 0 0 0 110.78 110.78l31-37.89A17 17 0 0 1 299 291a17.85 17.85 0 0 1 5.91 1.21l70 30A16.25 16.25 0 0 1 384 336a17.41 17.41 0 0 1-.39 3.37z&quot;></path></svg>
#                           </span>
#                           <span>+52 458 3215</span>
#                       </a>
#                   </li>
#                   <li>
#                       <a href=&quot;#!&quot; class=&quot;social-link&quot;>
#                           <span class=&quot;icon&quot;>
#                               <svg aria-hidden=&quot;true&quot; focusable=&quot;false&quot; data-prefix=&quot;fab&quot; data-icon=&quot;whatsapp&quot; class=&quot;svg-inline--fa fa-whatsapp fa-w-14&quot; role=&quot;img&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot; viewBox=&quot;0 0 448 512&quot;><path fill=&quot;currentColor&quot; d=&quot;M380.9 97.1C339 55.1 283.2 32 223.9 32c-122.4 0-222 99.6-222 222 0 39.1 10.2 77.3 29.6 111L0 480l117.7-30.9c32.4 17.7 68.9 27 106.1 27h.1c122.3 0 224.1-99.6 224.1-222 0-59.3-25.2-115-67.1-157zm-157 341.6c-33.2 0-65.7-8.9-94-25.7l-6.7-4-69.8 18.3L72 359.2l-4.4-7c-18.5-29.4-28.2-63.3-28.2-98.2 0-101.7 82.8-184.5 184.6-184.5 49.3 0 95.6 19.2 130.4 54.1 34.8 34.9 56.2 81.2 56.1 130.5 0 101.8-84.9 184.6-186.6 184.6zm101.2-138.2c-5.5-2.8-32.8-16.2-37.9-18-5.1-1.9-8.8-2.8-12.5 2.8-3.7 5.6-14.3 18-17.6 21.8-3.2 3.7-6.5 4.2-12 1.4-32.6-16.3-54-29.1-75.5-66-5.7-9.8 5.7-9.1 16.3-30.3 1.8-3.7.9-6.9-.5-9.7-1.4-2.8-12.5-30.1-17.1-41.2-4.5-10.8-9.1-9.3-12.5-9.5-3.2-.2-6.9-.2-10.6-.2-3.7 0-9.7 1.4-14.8 6.9-5.1 5.6-19.4 19-19.4 46.3 0 27.3 19.9 53.7 22.6 57.4 2.8 3.7 39.1 59.7 94.8 83.8 35.2 15.2 49 16.5 66.6 13.9 10.7-1.6 32.8-13.4 37.4-26.4 4.6-13 4.6-24.1 3.2-26.4-1.3-2.5-5-3.9-10.5-6.6z&quot;></path></svg>
#                           </span>
#                           <span>+52 458 8545</span>
#                       </a>
#                   </li>
#                   <li>
#                       <a href=&quot;#!&quot; class=&quot;social-link&quot;>
#                           <span class=&quot;icon&quot;>
#                               <svg aria-hidden=&quot;true&quot; focusable=&quot;false&quot; data-prefix=&quot;fas&quot; data-icon=&quot;envelope&quot; class=&quot;svg-inline--fa fa-envelope fa-w-16&quot; role=&quot;img&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot; viewBox=&quot;0 0 512 512&quot;><path fill=&quot;currentColor&quot; d=&quot;M502.3 190.8c3.9-3.1 9.7-.2 9.7 4.7V400c0 26.5-21.5 48-48 48H48c-26.5 0-48-21.5-48-48V195.6c0-5 5.7-7.8 9.7-4.7 22.4 17.4 52.1 39.5 154.1 113.6 21.1 15.4 56.7 47.8 92.2 47.6 35.7.3 72-32.8 92.3-47.6 102-74.1 131.6-96.3 154-113.7zM256 320c23.2.4 56.6-29.2 73.4-41.4 132.7-96.3 142.8-104.7 173.4-128.7 5.8-4.5 9.2-11.5 9.2-18.9v-19c0-26.5-21.5-48-48-48H48C21.5 64 0 85.5 0 112v19c0 7.4 3.4 14.3 9.2 18.9 30.6 23.9 40.7 32.4 173.4 128.7 16.8 12.2 50.2 41.8 73.4 41.4z&quot;></path></svg>
#                           </span>
#                           <span>info@smile.yes</span>
#                       </a>
#                   </li>
#               </ul>
#               <p>
#                   <a href=&quot;#!&quot; class=&quot;social-link&quot;>
#                           <span class=&quot;icon&quot;>
#                               <svg aria-hidden=&quot;true&quot; focusable=&quot;false&quot; data-prefix=&quot;fab&quot; data-icon=&quot;facebook&quot; class=&quot;svg-inline--fa fa-facebook fa-w-16&quot; role=&quot;img&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot; viewBox=&quot;0 0 512 512&quot;><path fill=&quot;currentColor&quot; d=&quot;M504 256C504 119 393 8 256 8S8 119 8 256c0 123.78 90.69 226.38 209.25 245V327.69h-63V256h63v-54.64c0-62.15 37-96.48 93.67-96.48 27.14 0 55.52 4.84 55.52 4.84v61h-31.28c-30.8 0-40.41 19.12-40.41 38.73V256h68.78l-11 71.69h-57.78V501C413.31 482.38 504 379.78 504 256z&quot;></path></svg>
#                           </span>
#                           <span>Facebook</span>
#                       </a>
#                   <a href=&quot;&quot; class=&quot;social-link&quot;>
#                       <span class=&quot;icon&quot;>
#                           <svg aria-hidden=&quot;true&quot; focusable=&quot;false&quot; data-prefix=&quot;fab&quot; data-icon=&quot;twitter&quot; class=&quot;svg-inline--fa fa-twitter fa-w-16&quot; role=&quot;img&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot; viewBox=&quot;0 0 512 512&quot;><path fill=&quot;currentColor&quot; d=&quot;M459.37 151.716c.325 4.548.325 9.097.325 13.645 0 138.72-105.583 298.558-298.558 298.558-59.452 0-114.68-17.219-161.137-47.106 8.447.974 16.568 1.299 25.34 1.299 49.055 0 94.213-16.568 130.274-44.832-46.132-.975-84.792-31.188-98.112-72.772 6.498.974 12.995 1.624 19.818 1.624 9.421 0 18.843-1.3 27.614-3.573-48.081-9.747-84.143-51.98-84.143-102.985v-1.299c13.969 7.797 30.214 12.67 47.431 13.319-28.264-18.843-46.781-51.005-46.781-87.391 0-19.492 5.197-37.36 14.294-52.954 51.655 63.675 129.3 105.258 216.365 109.807-1.624-7.797-2.599-15.918-2.599-24.04 0-57.828 46.782-104.934 104.934-104.934 30.213 0 57.502 12.67 76.67 33.137 23.715-4.548 46.456-13.32 66.599-25.34-7.798 24.366-24.366 44.833-46.132 57.827 21.117-2.273 41.584-8.122 60.426-16.243-14.292 20.791-32.161 39.308-52.628 54.253z&quot;></path></svg>
#                       </span>
#                       <span>Twitter</span>
#                   </a>
#                   <a href=&quot;&quot; class=&quot;social-link&quot;>
#                       <span class=&quot;icon&quot;>
#                           <svg aria-hidden=&quot;true&quot; focusable=&quot;false&quot; data-prefix=&quot;fab&quot; data-icon=&quot;instagram&quot; class=&quot;svg-inline--fa fa-instagram fa-w-14&quot; role=&quot;img&quot; xmlns=&quot;http://www.w3.org/2000/svg&quot; viewBox=&quot;0 0 448 512&quot;><path fill=&quot;currentColor&quot; d=&quot;M224.1 141c-63.6 0-114.9 51.3-114.9 114.9s51.3 114.9 114.9 114.9S339 319.5 339 255.9 287.7 141 224.1 141zm0 189.6c-41.1 0-74.7-33.5-74.7-74.7s33.5-74.7 74.7-74.7 74.7 33.5 74.7 74.7-33.6 74.7-74.7 74.7zm146.4-194.3c0 14.9-12 26.8-26.8 26.8-14.9 0-26.8-12-26.8-26.8s12-26.8 26.8-26.8 26.8 12 26.8 26.8zm76.1 27.2c-1.7-35.9-9.9-67.7-36.2-93.9-26.2-26.2-58-34.4-93.9-36.2-37-2.1-147.9-2.1-184.9 0-35.8 1.7-67.6 9.9-93.9 36.1s-34.4 58-36.2 93.9c-2.1 37-2.1 147.9 0 184.9 1.7 35.9 9.9 67.7 36.2 93.9s58 34.4 93.9 36.2c37 2.1 147.9 2.1 184.9 0 35.9-1.7 67.7-9.9 93.9-36.2 26.2-26.2 34.4-58 36.2-93.9 2.1-37 2.1-147.8 0-184.8zM398.8 388c-7.8 19.6-22.9 34.7-42.6 42.6-29.5 11.7-99.5 9-132.1 9s-102.7 2.6-132.1-9c-19.6-7.8-34.7-22.9-42.6-42.6-11.7-29.5-9-99.5-9-132.1s-2.6-102.7 9-132.1c7.8-19.6 22.9-34.7 42.6-42.6 29.5-11.7 99.5-9 132.1-9s102.7-2.6 132.1 9c19.6 7.8 34.7 22.9 42.6 42.6 11.7 29.5 9 99.5 9 132.1s2.7 102.7-9 132.1z&quot;></path></svg>
#                       </span>
#                       <span>Instagram</span>
#                   </a>
#               </p>
#           </footer>
#               <script src=&quot;https://cpwebassets.codepen.io/assets/common/stopExecutionOnTimeout-2c7831bb44f98c1391d6a4ffda0e1fd302503391ca806e7fcc7b9b87197aec26.js&quot;></script>
          
#             <script src='https://assets.calendly.com/assets/external/widget.js'></script>
#           <script src='https://unpkg.com/scrollreveal'></script>
#                 <script id=&quot;rendered-js&quot; >
#           'use strict';
          
#           const $body = document.body;
#           const $menuBtn = document.querySelector('#menubtn');
#           const $menuLbl = document.querySelector('.menulbl');
          
          
#           document.addEventListener('DOMContentLoaded', () => {
#             $body.addEventListener('click', e => {
#               const compare = e.target !== $menuBtn &amp;&amp; e.target !== $menuLbl;
#               compare &amp;&amp; ($menuBtn.checked = false);
#             });
#           });
          
#           // Scroll Reveal
#           const sr = ScrollReveal({
#             distance: '30px',
#             duration: 1800 });
          
          
#           sr.reveal(`.service:nth-child(2)`, {
#             origin: 'top',
#             interval: 200,
#             reset: true });
          
#           sr.reveal(`.service:nth-child(1), .service:nth-child(3)`);
          
#           sr.reveal(`.header__col:first-child, .info-element`, {
#             origin: 'left' });
          
#           sr.reveal(`.image`, {
#             origin: 'right' });
          
#           sr.reveal(`.testimonials`);
#           //# sourceURL=pen.js
#               </script>
#           </body>
          
#           </html>
#           " sandbox="allow-forms allow-modals allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-top-navigation-by-user-activation allow-downloads allow-presentation" allow="accelerometer; camera; encrypted-media; display-capture; geolocation; gyroscope; microphone; midi; clipboard-read; clipboard-write; web-share" allowtransparency="true" allowpaymentrequest="true" allowfullscreen="true" class="result-iframe">
#                 </iframe>
#             </div>
#           <div id="torrent-scanner-popup" style="display: none;"></div></body>
#           </html>
#     """ ,
#     unsafe_allow_html=True ,
# )
