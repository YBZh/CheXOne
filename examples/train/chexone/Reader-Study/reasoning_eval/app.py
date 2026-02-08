import ast
import json
import os
import re
from datetime import datetime
from itertools import count

import pandas as pd
import pymongo
import hashlib
import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import streamlit_survey as ss
import yaml
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from yaml.loader import SafeLoader

# Streamlit app session variables
st.set_page_config(layout="wide")
if 'initialize' not in st.session_state:
    st.session_state.initialize = True
if 'edit_durations' not in st.session_state:
    st.session_state["edit_durations"] = []
if 'next' not in st.session_state:
    st.session_state['next'] = False
if 'edit_completed' not in st.session_state:
    st.session_state['edit_completed'] = False
if "edflg" not in st.session_state:
    st.session_state.edflg = False
if "edited" not in st.session_state:
    st.session_state['edited'] = []
if "previously_edited" not in st.session_state:
    st.session_state['previously_edited'] = []
if "df_reader" not in st.session_state:
    st.session_state['df_reader'] = None


# # Initialize connection
# # Uses st.cache_resource to only run once
# @st.cache_resource
# def init_connection():
#     uri = st.secrets['mongo']['uri']
#     return MongoClient(uri, server_api=ServerApi('1'))
# client = init_connection()

MONGO_URI = None

# import urllib.parse

# username = urllib.parse.quote_plus("Yabin Zhang")
# password = urllib.parse.quote_plus("Mongodbyabin")
# MONGO_URI_QUOTED = f"mongodb+srv://{username}:{password}@cluster0.mongodb.net/?retryWrites=true&w=majority"
client = None



# client = None


def get_data(reader_id):
    # Load locally cached responses per reader
    responses_dir = 'responses'
    os.makedirs(responses_dir, exist_ok=True)
    file_path = os.path.join(responses_dir, f"reader-{reader_id}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to read local responses for reader {reader_id}: {e}")
            return {}
    return {}


# Authenticate
with open('readers.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)
def _render_login():
    try:
        authenticator.login(location='main')
    except Exception:
        try:
            authenticator.login('Login', 'main')
        except Exception:
            try:
                authenticator.login('main')
            except Exception:
                authenticator.login()
_render_login()
# Prefer reading values from Streamlit session state for maximum compatibility
name = st.session_state.get('name')
authentication_status = st.session_state.get('authentication_status')
username = st.session_state.get('username')
if authentication_status:
    st.write(f'Welcome to the CheXOne Reader Study on Reasoning Evaluation!')
elif authentication_status == False:
    st.error('Username or password is incorrect!')
elif authentication_status == None:
    st.warning('Please enter your provided username and password.')

# Data directories
data_folder = "raw_data/"

# Set reader ID
if username:
    reader_id = config['credentials']['usernames'][username]['id']
else:
    reader_id = None

# Start survey
if reader_id:
    survey = ss.StreamlitSurvey()
    try:
        # Load previous responses on initialization
        if st.session_state.initialize:
            print('Looking for previous responses...')
            item = get_data(reader_id)
            if item:
                print('Previous responses found.')
                # Detect new per-sample result format (top-level keys are sample_ids)
                is_new_format = isinstance(item, dict) and 'edited' not in item and 'edit_durations' not in item
                if is_new_format:
                    previously_edited = list(item.keys())
                    st.session_state['previously_edited'] = previously_edited
                    print(f"Previously edited samples: {previously_edited}")
                    # Do not attempt to restore widget states from file in new format
                else:
                    previously_edited = item.get('edited', [])
                    st.session_state['previously_edited'] = previously_edited
                    print(f"Previously edited samples: {previously_edited}")
                    item.pop('_id', None)
                    st.session_state["edit_durations"] = item.pop('edit_durations', [])
                    st.session_state["edited"] = item.pop('edited', [])
                    os.makedirs('temp', exist_ok=True)
                    with open(f"temp/{str(reader_id)}.json", 'w') as f:
                        json.dump(item, f)
                    survey.from_json(path=f"temp/{str(reader_id)}.json")

                # Filter out the samples that have already been submitted
                df_reader = pd.read_csv(os.path.join(data_folder, f"{reader_id}.csv"))
                if 'unique_id' in df_reader.columns:
                    df_reader = df_reader[~df_reader['unique_id'].isin(st.session_state.previously_edited)]
                else:
                    # Backward compatibility if only study_id exists
                    df_reader = df_reader[~df_reader['study_id'].isin(st.session_state.previously_edited)]
                st.session_state['df_reader'] = df_reader
            else:
                print('No previous responses found.')
                df_reader = pd.read_csv(os.path.join(data_folder, f"{reader_id}.csv"))
                st.session_state['df_reader'] = df_reader
            st.session_state.initialize = False
        else:
            df_reader = st.session_state.df_reader
    except Exception as e:
        print('Error loading previous responses.')
        print(e)
        df_reader = st.session_state.df_reader
        survey = ss.StreamlitSurvey()
        st.session_state.initialize = False

    # import ipdb; ipdb.set_trace()
    # Format data as dicts (handle missing columns gracefully)
    if 'unique_id' in df_reader.columns:
        df_reader = df_reader.sort_values('unique_id')
    reader_dict_questions = (
        df_reader.set_index('unique_id')['question'].to_dict()
        if 'question' in df_reader.columns
        else {}
    )
    reader_dict_reports = (
        df_reader.set_index('unique_id')['candidate'].to_dict()
        if 'candidate' in df_reader.columns
        else {}
    )
    reader_dict_labels = (
        df_reader.set_index('unique_id')['labels'].to_dict()
        if 'labels' in df_reader.columns
        else {}
    )
    reader_dict_reasoning = df_reader.set_index('unique_id')['reasoning'].to_dict()
    reader_dict_images = df_reader.set_index('unique_id')['image_paths'].to_dict()
    reader_dict_task = (
        df_reader.set_index('unique_id')['task_name'].to_dict()
        if 'task_name' in df_reader.columns
        else {}
    )

    def get_ab_sources(sample_id):
        # Deterministic blind assignment for A/B per sample
        h = hashlib.md5(str(sample_id).encode('utf-8')).hexdigest()
        if int(h, 16) % 2 == 0:
            return ('labels', 'candidate')
        else:
            return ('candidate', 'labels')

    # 新数据格式，按字典（key=unique_id, value=图像列表字符串）
    for each_sample in reader_dict_images:
        # 取出该样本的 images 字符串
        image_paths_str = reader_dict_images[each_sample]
        # 安全地反序列化为python列表
        try:
            image_path_list = ast.literal_eval(image_paths_str)
        except Exception as e:
            print(f"Error parsing image paths for {each_sample}: {image_paths_str}")
            image_path_list = []

        # 自动命名View: 'View 1', 'View 2', ... 不再考虑'nan'问题
        sample_images_dict = {}
        for idx, img_path in enumerate(image_path_list):
            key = f"View {idx+1}"
            # 若图片已存同名，自动避免冲突
            new_key = key
            counter = 1
            while new_key in sample_images_dict:
                new_key = f"{key} ({counter})"
                counter += 1
            sample_images_dict[new_key] = img_path
        reader_dict_images[each_sample] = sample_images_dict
    # import ipdb; ipdb.set_trace()
    # Number of samples
    study_ids = list(reader_dict_reports.keys())
    num_samples = len(study_ids)

    print('\n-----------------------------------')
    print(f"Reader ID: {reader_id}")
    # print(f"Sample IDs: {study_ids}")
    print(f"Number of samples: {num_samples}")
    print('-----------------------------------\n')


    # Save function to persist current sample only
    def save(survey, reader_id=reader_id, sample_id=None):
        if sample_id is None:
            return
        json_file = json.loads(survey.to_json())
        # Extract only factuality and causal support fields for this sample
        sample_payload = {
            "factuality": None,
            "causal_support": None,
        }
        # Keys are like: "factuality_reader-{reader_id}_{sample_id}" etc.
        suffix = f"reader-{reader_id}_{sample_id}"
        for k, v in json_file.items():
            if isinstance(v, dict) and k.endswith(suffix):
                key_lower = k.lower()
                value = v.get("value")
                if key_lower.startswith("factuality_reader-"):
                    sample_payload["factuality"] = value
                elif key_lower.startswith("causal_support_reader-"):
                    sample_payload["causal_support"] = value

        # Find duration for this sample from session durations
        duration_value = None
        for entry in st.session_state.get('edit_durations', []):
            if isinstance(entry, dict) and entry.get("id") == f"time_reader-{reader_id}_{sample_id}":
                # entry['duration'] is already seconds (float)
                duration_value = entry.get("duration")
                break
        if duration_value is not None:
            sample_payload["duration"] = duration_value

        # Persist per user locally as a dict of sample_id -> sample_payload
        responses_dir = 'responses'
        os.makedirs(responses_dir, exist_ok=True)
        file_path = os.path.join(responses_dir, f"reader-{reader_id}.json")
        existing = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as rf:
                    existing = json.load(rf) or {}
            except Exception:
                existing = {}
        existing[sample_id] = sample_payload
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"Saved locally to {file_path}")

    # Stop if all samples have been reviewed
    if len(study_ids) == 0:
        st.error("No samples to review.")
        st.stop()

    # If survey has been completed, show a prominent banner and stop
    if st.session_state.get('survey_completed'):
        st.markdown(
            """
            <div style="text-align:center; padding: 24px; border-radius: 8px; background-color:#e6ffed; border:1px solid #b7eb8f; font-size:20px; font-weight:600;">
            All responses have been recorded. You may close the window. Thank you!
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            st.balloons()
        except Exception:
            pass
        st.stop()

    # Survey pages
    def _on_submit():
        st.session_state['survey_completed'] = True
    pages = survey.pages(num_samples, on_submit=_on_submit)
    with pages:
        # Study-level totals (include resumed/completed samples in totals)
        responses_path = os.path.join('responses', f"reader-{reader_id}.json")
        saved_dict = {}
        try:
            if os.path.exists(responses_path):
                with open(responses_path, "r", encoding="utf-8") as rf:
                    saved_dict = json.load(rf) or {}
        except Exception:
            saved_dict = {}

        total_studies_all = 0
        reviewed_studies = 0
        try:
            df_all_reader = pd.read_csv(os.path.join(data_folder, f"{reader_id}.csv"))
            all_study_ids = [str(uid) for uid in df_all_reader.get('unique_id', []).tolist()]
            total_studies_all = len(all_study_ids)
            saved_ids = set(saved_dict.keys()) if isinstance(saved_dict, dict) else set()
            reviewed_studies = len(saved_ids & set(all_study_ids))
        except Exception:
            total_studies_all = 0
            reviewed_studies = 0
        remaining_studies = max(total_studies_all - reviewed_studies, 0)
        st.markdown(f"**Studies — Total:** {total_studies_all}  |  **Reviewed:** {reviewed_studies}  |  **Remaining:** {remaining_studies}")
        try:
            st.progress(0.0 if total_studies_all == 0 else reviewed_studies / total_studies_all)
        except Exception:
            pass

        sample = study_ids[pages.current]
        images = reader_dict_images[sample]


        # Page buttons (Next, Submit, Previous)
        def next_sample():
            pages.next()
            st.session_state.pop('started', None)
            st.session_state.next = False
            st.session_state.edit_completed = False
            st.session_state.edflg = False


        pages.next_button = lambda pages: st.button(
            "Next",
            type="primary",
            use_container_width=True,
            on_click=next_sample,
            disabled=pages.current == pages.n_pages - 1,
            key=f"{pages.current}_btn_next",
        ) if st.session_state['next'] else None
        pages.submit_button = lambda pages: st.button(
            "Submit",
            type="primary",
            use_container_width=True,
        ) if st.session_state['next'] else None
        pages.prev_button = lambda pages: None

        # Start time
        if 'started' not in st.session_state:
            print(f"Now viewing sample {pages.current}: {sample}")
            st.session_state[f"start_edit_time_reader-{reader_id}_{sample}"] = datetime.now()
            st.session_state.started = True

        # Page content
        left, right = st.columns([0.6, 0.4])
        with left:
            """#### 1. Review DICOM:"""
            # st.write(
            #     f"**Tips:** Scroll to Zoom, Scroll Click/Drag to Pan, Left Click/Drag to adjust Constrast (Left/Right) and Brightness (Up/Down)."
            # )
            st.write(
                f"**Tips:** There may be multiple views of the same study. Please review all views."
            )

            options = st.radio(
                '**DICOM View:**',
                list(images.keys()),
                horizontal=True,
                key=f"radio_reader-{reader_id}_{sample}"
            )

            img_file = images[options]
            # print(img_file)

            # Rewrite to point to the filtered_images_clip_100_random_seed42/data bucket/object path
            # img_file is the filename only, sample is the unique_id,
            # In this dataset, the path is:
            # https://storage.googleapis.com/reader_study/filtered_images_clip_100_random_seed42/data/[REST_OF_PATH]
            # The csv stores full relative paths under image_paths, e.g.
            # './data/mimic-cxr/files/p17/p17669276/s52816124/107bf819-bd17b10b-9fa1cd26-692e07cc-b408328a.jpg'
            # We need the path under 'data/...'
            # So we drop the leading './data/' from the relative path

            # If img_file starts with './data/', strip that
            if img_file.startswith('./data/'):
                img_gcs_path = img_file[len('./data/'):]
            elif img_file.startswith('data/'):
                img_gcs_path = img_file[len('data/'):]
            else:
                img_gcs_path = img_file
            print(img_gcs_path)
            # import ipdb; ipdb.set_trace()
            img_url = f"https://storage.googleapis.com/readerstudychexone/reasoneval_rexvqa/data/{img_gcs_path}"

            # 尝试直接展示图片，如果是jpg/png而不是dicom文件
            if img_url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                st.image(img_url, width=700, caption=f"Image: {options}")
            else:
                # DICOM 需要cornerstone或外部viewer支持，这里简单给出外部链接
                st.markdown(
                    f"[打开DICOM Viewer查看]({img_url})\n\n"
                    "（如不能自动在页面内展示，可点击此链接在新窗口打开）"
                )
                cornerstone_react_component_url = (
                    f"https://vilmedic.app/study/NLG?width=800&height=800&fileUrl={img_url}"
                )
                # 仍然尝试iframe嵌入，但提示可能需要新窗口
                components.iframe(cornerstone_react_component_url, width=700, height=700)
        with right:
            """#### 2. Review Question and Reasoning:"""

            st.write('**Question:**')
            _q = reader_dict_questions.get(sample, "")
            if isinstance(_q, str) and _q:
                # Normalize newlines and force markdown line breaks
                _q = _q.replace('\r\n', '\n').replace('\r', '\n')
                st.markdown(_q.replace('\n', '  \n'))
            else:
                st.write("")
            # if reader_dict_task:
            #     st.write('**Task:**')
            #     st.write(reader_dict_task.get(sample, ""))
            st.write('**Reasoning:**')
            st.write(reader_dict_reasoning.get(sample, ""))
            st.write('**Answer:**')
            st.write(reader_dict_reports.get(sample, ""))

            """#### 3. Factuality:"""
            factuality_key = f"factuality_reader-{reader_id}_{sample}"
            factuality_selected = survey.selectbox(
                '**Factuality: The reasoning section correctly describes chest X-ray findings relevant to the images**',
                options=["", "Strongly Agree", "Agree", "Neutral/Not Applicable", "Disagree", "Strongly Disagree"],
                index=0,
                id=factuality_key
            )
            """#### 3. Causal Support:"""
            causal_support_key = f"causal_support_reader-{reader_id}_{sample}"
            causal_support_selected = survey.selectbox(
                '**Causal Support: The reasoning section logically and causally supports the final answer.**',
                options=["", "Strongly Agree", "Agree", "Neutral/Not Applicable", "Disagree", "Strongly Disagree"],
                index=0,
                id=causal_support_key
            )

            # Only enable SAVE RESPONSES button if selection is made
            all_selected = (factuality_selected != "" and causal_support_selected != "")
            def enable_next():
                if not all_selected:
                    st.warning("请先完成所有必选项。")
                    return
                save(survey, sample_id=sample)
                st.session_state['next'] = True


            if st.button("SAVE RESPONSES", key=f"feedback_save_reader-{reader_id}_{sample}",
                            use_container_width=True, on_click=enable_next, disabled=not all_selected):
                st.success(f"Feedback saved!")