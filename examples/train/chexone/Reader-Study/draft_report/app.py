import ast
import json
import os
import re
from datetime import datetime
from itertools import count

import pandas as pd
import pymongo
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
    st.write(f'Welcome to the CheXOne Reader Study on Report Generation!')
elif authentication_status == False:
    st.error('Username or password is incorrect!')
elif authentication_status == None:
    st.warning('Please enter your provided username and password.')

# Data directories
data_folder = "raw_data_mimic_report/"

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
    # Format data as dicts
    df_reader.sort_values('unique_id')
    reader_dict_reports = df_reader.set_index('unique_id')['candidate'].to_dict()
    reader_dict_indications = df_reader.set_index('unique_id')['indication'].to_dict()
    reader_dict_images = df_reader.set_index('unique_id')['image_paths'].to_dict()

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
        # Persist duration and the free text ("Write") for this sample
        json_file = json.loads(survey.to_json())
        # Helper to normalize labels
        def _clean_label(lbl: str) -> str:
            lbl = lbl.strip()
            if lbl.startswith("**") and lbl.endswith("**"):
                lbl = lbl.strip("*").strip()
            return lbl

        duration_value = None
        write_text_value = None

        # Find duration for this sample from session durations
        for entry in st.session_state.get('edit_durations', []):
            if isinstance(entry, dict) and entry.get("id") == f"time_reader-{reader_id}_{sample_id}":
                duration_value = entry.get("duration")
                break

        # Extract only the "Write" text input for this sample
        suffix = f"reader-{reader_id}_{sample_id}"
        for k, v in json_file.items():
            if isinstance(v, dict) and k.endswith(suffix):
                label = _clean_label(v.get("label", ""))
                # Remove trailing colon for cleanliness
                key = label[:-1] if label.endswith(":") else label
                if key == "Write":
                    write_text_value = v.get("value")

        sample_payload = {}
        if duration_value is not None:
            sample_payload["duration"] = duration_value
        if write_text_value is not None:
            sample_payload["Write"] = write_text_value

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
            """#### 1. CXR Image:"""
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
            img_url = f"https://storage.googleapis.com/readerstudychexone/reportgen/data/{img_gcs_path}"

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
            """#### 2. Draft a Finding Section for This Study (Timer Running):"""
            st.write('**Exam Indication:**')
            st.write(reader_dict_indications[sample])
            # st.write('**Report (from a Model or a Radiologist):**')

            # # Display report
            # st.write(reader_dict_reports[sample])

            # 只保留write from scratch
            survey.text_area(
                "**Write**",
                value="",
                id=f"edit_reader-{reader_id}_{sample}",
                height=150,
                disabled=st.session_state.edflg
            )


            def disable_edit():
                st.session_state.edflg = True


            # Submit process
            if st.button("SUBMIT", key=f"edit_save_reader-{reader_id}_{sample}", on_click=disable_edit,
                         use_container_width=True, disabled=st.session_state.edflg):
                st.success(f"Submitted!")
                st.session_state[f"end_edit_time_reader-{reader_id}_{sample}"] = datetime.now()
                time = {
                    "id": f"time_reader-{reader_id}_{sample}",
                    "start": st.session_state[f"start_edit_time_reader-{reader_id}_{sample}"],
                    "end": st.session_state[f"end_edit_time_reader-{reader_id}_{sample}"],
                    "duration": st.session_state[f"end_edit_time_reader-{reader_id}_{sample}"] - st.session_state[
                        f"start_edit_time_reader-{reader_id}_{sample}"]
                }
                time['duration'] = time['duration'].total_seconds()
                st.session_state[f"edit_durations"].append(time)
                st.session_state['edited'].append(sample)
                st.session_state['edit_completed'] = True
                # Persist only duration and enable Next
                save(survey, sample_id=sample)
                st.session_state['next'] = True

            # # Show feedback form if edit is completed
            # if st.session_state['edit_completed']:
            #     """#### 3. Log Feedback:"""
            #     import streamlit as st

            #     # 3.1
            #     why_key = f"why_reader-{reader_id}_{sample}"
            #     why_options = [
            #         "",
            #         "No editing needed (good report)",
            #         "[Content] False / Missing report of a finding in the image",
            #         "[Style] Poor report writing style",
            #         "Both content and style need improvement"
            #     ]
            #     why_selected = survey.selectbox(
            #         '**3.1: Why did you make those edits?** (Required)',
            #         options=why_options,
            #         index=0,
            #         id=why_key
            #     )
            #     # 3.2
            #     indication_key = f"indication_reader-{reader_id}_{sample}"
            #     indication_options = [
            #         "",
            #         "Strongly Agree",
            #         "Agree", 
            #         "Neutral/Not Applicable", 
            #         "Disagree", 
            #         "Strongly Disagree", 
            #     ]
            #     indication_selected = survey.selectbox(
            #         "**3.2: Applicability to exam indication:  The drafted report helps answer the exam indication:** (Required)", 
            #         options=indication_options,
            #         index=0,
            #         id=indication_key
            #     )
            #     # 3.3
            #     writing_efficiency_key = f"writing_efficiency_reader-{reader_id}_{sample}"
            #     writing_efficiency_selected = survey.selectbox(
            #         "**3.3: Writing efficiency:  The drafted report improves report writing efficiency:** (Required)", 
            #         options=indication_options,
            #         index=0,
            #         id=writing_efficiency_key
            #     )
            #     # 3.4
            #     interpretation_efficiency_key = f"interpretation_efficiency_reader-{reader_id}_{sample}"
            #     interpretation_efficiency_selected = survey.selectbox(
            #         "**3.4: Interpretation efficiency:  The drafted report improves CXR interpretation efficiency:** (Required)", 
            #         options=indication_options,
            #         index=0,
            #         id=interpretation_efficiency_key
            #     )

            #     # Only enable SAVE RESPONSES button if all fields have valid (non-empty) values
            #     all_selected = (
            #         why_selected != "" and
            #         indication_selected != "" and
            #         writing_efficiency_selected != "" and
            #         interpretation_efficiency_selected != ""
            #     )
            #     # import ipdb; ipdb.set_trace()
            #     # print(f"Saving feedback for {survey}")
            #     def enable_next():
            #         if not all_selected:
            #             st.warning("请先完成所有必选项。")
            #             return
            #         save(sample_id=sample)
            #         st.session_state['next'] = True


            #     if st.button("SAVE RESPONSES", key=f"feedback_save_reader-{reader_id}_{sample}",
            #                  use_container_width=True, on_click=enable_next, disabled=not all_selected):
            #         st.success(f"Feedback saved!")