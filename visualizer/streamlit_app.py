from .parse_and_vis import *

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    argumentparser.parse()
    setup_globals()
    if not st.session_state.show_filepicker:
        start_newfile = st.button("Profile another file")
        if start_newfile:
            st.session_state.show_filepicker = True

    if st.session_state.show_filepicker:
        filepicker_page.filepicker_page()
    else:
        continue_main()
    # print(clicked_src)