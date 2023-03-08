import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import os
import base64
import pickle
import tracktor as tr
import sys
import cv2 
import uuid
from PIL import Image
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from Bio import SeqUtils
from Bio.SeqUtils import ProtParam
import io
import subprocess
import matplotlib.pyplot as plt

# Molecular descriptor calculator
#def desc_calc():
#    # Performs the descriptor calculation
#    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
#    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#    output, error = process.communicate()
#    os.remove('molecule.smi')

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">下载预测结果</a>'
    return href

# Model building
def build_model(input_data):
    # Reads in saved regression model
    load_model = pickle.load(open('tox_model.pkl', 'rb'))
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header('**鱼毒性预测值如下**')
    prediction_output = pd.Series(prediction, name='pLC50')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.dataframe(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

    st.session_state['prediction'] = df




# Logo image
image = Image.open('logo.png')

st.image(image, use_column_width=True)





# Page title
st.markdown("""
<h1 style='text-align: center'>⚡欢迎来到小闪电实验室⚡</h1>
<h4 style='text-align: center'>请在侧边栏选择你要运行的程序</h4>
""", unsafe_allow_html=True)

# Sidebar
#image1 = cv2.imread('./dead-fish.png')
with st.sidebar.expander('农药鱼毒性预测'):
    #st.image(image1, use_column_width=True)
    example_file = 'example_acetylcholinesterase.txt'
    if st.button('下载示例文件'):
        with open(example_file, 'r') as f:
            text = f.read()
        b64 = base64.b64encode(text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{example_file}">点击这里下载</a>'
        st.markdown(href, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("或者上传你的数据", type=['txt'])

    if st.button('开始预测'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**原始输入数据**')
            st.dataframe(load_data)

            with st.spinner("计算分子描述符..."):
                desc_calc()    
                st.header('**计算好的分子描述符**')
                desc = pd.read_csv('descriptors_output.csv')
                st.dataframe(desc)
                st.write(desc.shape)

                # Read descriptor list used in previously built model
                st.header('**之前构建好的模型**')
                Xlist = list(pd.read_csv('descriptor_list.csv').columns)
                desc_subset = desc[Xlist]
                st.dataframe(desc_subset)
                st.write(desc_subset.shape)

                # Apply trained model to make prediction on query compounds
                build_model(desc_subset)
        else:
            st.warning('请上传一个数据文件')



unique_id = str(uuid.uuid4())



# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="protein_properties.csv">下载蛋白质理化性质预测结果</a>'
    return href

# Sidebar
#image2 = cv2.imread('./protein.jpeg')
with st.sidebar.expander('蛋白质理化性质预测'):
    #st.image(image2, use_column_width=True)
    example_file = 'protein_sequence_sample.csv'
    if st.button('下载示例文件', key='download_example_file'):
        df = pd.read_csv(example_file)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="{example_file}">点击这里下载</a>'
        st.markdown(href, unsafe_allow_html=True)


    uploaded_file = st.file_uploader("上传包含蛋白质序列的csv文件")

    if st.button('开始预测', key='predict_protein_properties'):

        if uploaded_file is not None:
            protein_data = pd.read_csv(uploaded_file)
            protein_seq = ''.join(protein_data['Protein sequence'][0].split())
            protein_name = protein_data['Protein name'][0]

            # Compute protein properties
            X = pd.DataFrame(columns=['Protein', 'Amino acid count', 'Molecular weight', 'PI', 'GRAVY',
                                      'instability index', 'Negative residue count', 'Positive residue count',
                                      'Predicted TM domain count', 'Predicted signal peptide count', 'Predicted phosphorylation site count'])
            p = ProtParam.ProteinAnalysis(protein_seq)
            ii = p.instability_index()
            aa_count = len(protein_seq)
            mw = p.molecular_weight()
            pi = p.isoelectric_point()
            # 计算氨基酸序列的总亲疏水性
            total_hydrophobicity = sum(SeqUtils.ProtParamData.kd[aa] for aa in protein_seq)
            # 计算氨基酸序列的GRAVY指数
            gravy = total_hydrophobicity / (mw - len(protein_seq))

            #aliphatic_index = SeqUtils.ProtParamData.aliphatic_index(protein_seq)
            neg_count = protein_seq.count('D') + protein_seq.count('E')
            pos_count = protein_seq.count('R') + protein_seq.count('K')
            tm_count = 0  # Predicted TM domain count, to be implemented
            signal_count = 0  # Predicted signal peptide count, to be implemented
            phos_count = 0  # Predicted phosphorylation site count, to be implemented

            X.loc[0] = [protein_name, aa_count, mw, pi, gravy, ii, neg_count, pos_count,
                        tm_count, signal_count, phos_count]

            st.header('**蛋白质理化性质预测结果**')
            st.dataframe(X)
            st.markdown(filedownload(X), unsafe_allow_html=True)

            st.session_state['prediction'] = X

        else:
            st.warning('请上传一个数据文件')


def detect_and_draw_contours(frame, thresh, meas_last, meas_now, min_area = 0, max_area = 10000):

    # Detect contours and draw them based on specified area thresholds
    if int(cv2.__version__[0]) == 3:
        img, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    final = frame.copy()

    i = 0
    meas_last = meas_now.copy()
    del meas_now[:]
    if len(contours) > 0:
        img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        while i < len(contours):
            area = cv2.contourArea(contours[i])
            if area < min_area or area > max_area:
                contours = contours[:i] + contours[i+1:]
            else:
                cv2.drawContours(final, contours, i, (0,0,255), 1)
                M = cv2.moments(contours[i])
                if M['m00'] != 0:
                    cx = M['m10']/M['m00']
                    cy = M['m01']/M['m00']
                else:
                    cx = 0
                    cy = 0
                meas_now.append([cx,cy])
                i += 1
    else:
        img = thresh
    return final, contours, meas_last, meas_now


# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())


n_inds = 5
t_id = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
colours = [(0,0,255),(0,255,255),(255,0,255),(255,255,255),(255,255,0),(255,0,0),(0,255,0),(0,0,0)]
block_size = 51
offset = 25
scaling = 1.0
min_area = 50
max_area = 1000
mot = True
codec = 'DIVX'

temp_file_to_save = './temp_file_1.mp4'
temp_file_result  = './temp_file_2.mp4'
output_filepath = './temp_file_3.csv'


image3 = cv2.imread('./zebrafish.jpeg')
with st.sidebar.expander('斑马鱼运动行为监测'):
    st.image(image3, use_column_width=True)
    st.download_button(label="下载示例视频", data=open('./斑马鱼.mp4', 'rb').read(), file_name='斑马鱼示例视频.mp4', mime='video/mp4')
    video_data = st.file_uploader("Upload file", ['mp4','mov', 'avi'])
    if video_data is not None:

        write_bytesio_to_file(temp_file_to_save, video_data)

        cap = cv2.VideoCapture(temp_file_to_save)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = cap.get(cv2.CAP_PROP_FPS)  
        st.write(width, height, frame_fps)
        

        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_framesize = (int(cap.read()[1].shape[1]*scaling),int(cap.read()[1].shape[0]*scaling))
        out_mp4 = cv2.VideoWriter(filename = temp_file_result, fourcc = fourcc, fps = 60.0, frameSize = output_framesize, isColor = True)

        meas_last = list(np.zeros((n_inds,2)))
        meas_now = list(np.zeros((n_inds,2)))
        df = []
        last = 0
        while True:
            ret, frame = cap.read()
            this = cap.get(1)
            if ret == True:

                frame = cv2.resize(frame, None, fx = scaling, fy = scaling, interpolation = cv2.INTER_LINEAR)
                thresh = tr.colour_to_thresh(frame, block_size, offset)
                final, contours, meas_last, meas_now = detect_and_draw_contours(frame, thresh, meas_last, meas_now, min_area, max_area)
                if len(meas_now) != n_inds:
                    contours, meas_now = tr.apply_k_means(contours, n_inds, meas_now)
                
                row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
                final, meas_now, df = tr.reorder_and_draw(final, colours, n_inds, col_ind, meas_now, df, mot, this)
                

                for i in range(n_inds):
                    df.append([this, meas_now[i][0], meas_now[i][1], t_id[i]])
                

                out_mp4.write(final)
                    
            if last >= this:
                break
            
            last = this
        
        ## Close video files
        out_mp4.release()
        cap.release()
        df = pd.DataFrame(np.matrix(df), columns = ['frame','pos_x','pos_y', 'id'])
        df.to_csv(output_filepath, sep=',')

        ## Reencodes video to H264 using ffmpeg
        ##  It calls ffmpeg back in a terminal so it fill fail without ffmpeg installed
        ##  ... and will probably fail in streamlit cloud
        convertedVideo = "./testh264.mp4"
        subprocess.call(args=f"ffmpeg -y -i {temp_file_result} -c:v libx264 {convertedVideo}".split(" "))
     
        ## Show results
        col1,col2 = st.columns(2)
        col1.header("原始视频")
        col1.video(temp_file_to_save)

        col2.header("分析处理后的视频")
        col2.video(convertedVideo)

        df = pd.read_csv(output_filepath)

        tmp = df[df['frame'] < 500]
        fig, ax = plt.subplots(figsize=(16,10), dpi=1200)
        ax.scatter(tmp['pos_x'], tmp['pos_y'], c=tmp.index, s=10)
        ax.set_xlabel('pos_x')
        ax.set_ylabel('pos_y')
        st.pyplot(fig)
        plt.savefig('./pos.jpeg')


        # calculate speed for each individual
        df['speed'] = np.sqrt((df['pos_x'].diff())**2 + (df['pos_y'].diff())**2)
        df['speed'] = df['speed'] / (1/60)  # convert speed to pixels per second

        # plot speed vs. time for each individual
        fig, ax = plt.subplots(figsize=(16,10), dpi=1200)
        for id in np.unique(df['id']):
            ax.plot(df[df['id']==id]['frame'], df[df['id']==id]['speed'], label='Individual ' + str(id))
        ax.set_xlabel('Frame')
        ax.set_ylabel('Speed (pixels/s)')
        ax.legend()
        st.pyplot(fig)
        plt.savefig('./speed.jpeg')


        st.write('分析结束，处理后的视频和图像：）')
        st.download_button(label="下载处理后的视频", data=open('./temp_file_2.mp4', 'rb').read(), file_name='处理后的视频.mp4', mime='video/mp4')
        st.download_button(label="下载斑马鱼坐标变化文件", data=open('./temp_file_3.csv', 'rb').read(), file_name='坐标变化.csv', mime='text/csv')
        st.download_button(label="下载斑马鱼游动轨迹图像", data=open('./pos.jpeg', 'rb').read(), file_name='轨迹.jpeg', mime='image/png')
        st.download_button(label="下载斑马鱼速度变化图像", data=open('./speed.jpeg', 'rb').read(), file_name='速度变化.jpeg', mime='image/png')













