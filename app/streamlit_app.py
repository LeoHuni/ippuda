# -*- coding: utf-8 -*-
import streamlit as st
import os
import cv2
from PIL import Image
from sys import path
import numpy as np

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


st.title('이뿌다 가상 성형 AI')

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
    mode = [ 'About App' , '가상 성형 AI' , '피부 진단 AI' ]
    app = st.sidebar.selectbox('choose the Appmode' ,
                                    mode ,
                                    key=str(num)
                                    )
    num += 1
    return app


app_mode = mode_select(count)

if app_mode == 'About App':
    st.markdown('In this App we are using Dlib for creating morphing App')
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

elif app_mode == '가상 성형 AI':
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

    max_faces = st.sidebar.number_input('Maximum Number of Pictures' , value=2 , min_value=1 , key='num_input')
    st.sidebar.markdown('---')
    CHANGE_GRADE = st.sidebar.slider('얼마나 바꿔볼래?' , min_value=0.0 , max_value=0.99 , value=0.5 , key='slider')
    st.sidebar.markdown('---')

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
    if src_image is not None:
        myImage = np.array(Image.open(src_image))
        st.sidebar.text('내 사진')
        st.sidebar.image(myImage)
    if src_target_image is not None:
        targetImage = np.array(Image.open(src_target_image))
        st.sidebar.text('워너비 사진')
        st.sidebar.image(targetImage)
    if st.button("가상 성형 시작 !" , key='morph_start'):
        with st.spinner('이뿌게 성형 중이에용 ~!!'):

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

            index = int(CHANGE_GRADE * 100)
            st.image(morph_array_origin[index])
            # image_res = Image.fromarray(morph_array[index])
            # st.image(image_res)
        st.success('성형 끗 !!')
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
        R_eye_res = round(res_tot[ 1 ] , 2)
        nose_bridge_res = round(res_tot[ 2 ] , 2)
        nos_res = round(res_tot[ 3 ] , 2)
        face_res = round(res_tot[ 4 ] , 2)
        L_eyebrow_res = round(res_tot[ 5 ] , 2)
        R_eyebrow_res = round(res_tot[ 6 ] , 2)
        U_mouth_res = round(res_tot[ 7 ] , 2)
        L_mouth_res = round(res_tot[ 8 ] , 2)

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

elif app_mode == '피부 진단 AI':
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




