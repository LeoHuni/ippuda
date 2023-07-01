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

from streamlit.components.v1 import html
from strealit.components.v1 as components
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
    mode = [ '🖐️   About App' , '💉   가상 성형 AI' , '👧   피부 진단 AI', '🏥   입점 병원', '⏰   병원 상세 & 예약',
           '🏠   UI 샘플']
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
        
    st.markdown('---')
    st.subheader("🚨 👷‍♀️👷‍♂️   공사중입니다   👷‍♂️👷‍♀️ 🚨")
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

        # html(my_html)
        # st.markdown(my_html , unsafe_allow_html=True)  # JavaScript doesn't work

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
        path_to_html ='app/src/info.html'
        # Read file and keep in variable
        with open(path_to_html,'r') as f: 
            html_data = f.read()
        st.components.v1.html(html_data,height=2000)

elif app_mode == '🏠   UI 샘플':
        path_html = 'app/src/figma-to-html/index.html'
        with open(path_html,'r') as s: 
            html_ui = s.read()
        st.components.v1.html(html_ui,height=2000)
