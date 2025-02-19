import numpy as np
import streamlit as st

def handle_reference_text_count_add():
    if st.session_state.reference_text_count < 3:
        st.session_state.reference_text_count += 1
    else:
        st.session_state.reference_text_count = 3
    
def handle_reference_text_count_remove():
    if st.session_state.reference_text_count > 1:
        st.session_state.reference_text_count -= 1
    else:
        st.session_state.reference_text_count = 1

def click_button():
    st.session_state.clicked = True

def build_ref_boxes():
    texts = {}
    height = 75
    if st.session_state.reference_text_count == 1:
        height = 315
    if st.session_state.reference_text_count == 2:
        height = 135
    
    for ref in range(0, st.session_state.reference_text_count):
        txt = st.text_area(f"Reference Text - {ref+1}", height=height)
        
        if len(txt) > 1:
            texts[f"Reference Text - {ref+1}"] = txt
    
    return texts

def main():

    if "reference_text_count" not in st.session_state:
        st.session_state['reference_text_count'] = 2
    
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    # Use the full page instead of a narrow central column
    st.set_page_config(layout="wide")

    # Space out the maps so the first one is 2x the size of the other three
    c1, c2 = st.columns((1, 1))

    with c1:
        txt = st.text_area("Input Text", height=315)
        compute_text_button = st.button('Compute', on_click=click_button)

    with c2:

        ref_texts = build_ref_boxes()

        b1, b2, b3 = st.columns((0.25, 0.5, 0.5), gap="small", border=False)
        with b1:
            add_text_button = st.button('Add', on_click=handle_reference_text_count_add)
        with b2:
            remove_text_button = st.button('Remove', on_click=handle_reference_text_count_remove)
        with b3:
            clear_text_button = st.button('Clear')
            if clear_text_button:
                ref_texts = {}
                st.session_state.clicked = False

    if st.session_state.clicked and len(ref_texts.values()) > 0:
        
        option = st.selectbox(label="Analysis", options=tuple(ref_texts.keys()))
    
        text_to_show = ref_texts.get(option)

        st.write(text_to_show)



if __name__ == "__main__":
    main()