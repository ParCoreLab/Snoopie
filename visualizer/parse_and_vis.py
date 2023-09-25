import sys
import math
import os
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import pickle
import extra_streamlit_components as stx
import streamlit.components.v1 as components
from st_click_detector import click_detector
import seaborn as sns
import pandas as pd
import numpy as np
import colorsys
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, ColumnsAutoSizeMode
import io
from includes import argumentparser, filepicker_page, electron_checker, filepath_handler
from includes.streamlit_globals import *
from includes.parser import *
from includes.tables import *
from st_clickable_images import clickable_images



data_by_line = {}
reverse_table_lineinfo = {}
src_lines = {}
chosen_line: None | str = None
file_to_vis: None | str = None
existing_src_code_files = {}
ops_by_line_info: Dict[str, Dict[int | str, int]] = {
    "by_line_index" : {}, "by_file": {}, "total": 0, "by_file_linenum": {}}
# addr_obj_map = {}

ops = set()
ops_to_display = []

GRAPH_SHAPES = {"square": 0, "polygon": 1, "rectangle": 2}
MIN_TABLE_HEIGHT = 50
ROW_HEIGHT = 27
MAX_TABLE_HEIGHT = 540

GRAPH_SIZE_LIMIT = 100000
ADDR_DIST = 4
colorscale = [[0.00000000, "rgb(230, 233, 233)"],
              [0.05555555, "rgb(209, 227, 172)"],
              [0.16666666, "rgb(186, 220, 124)"],
              [0.26666666, "rgb(143, 216, 115)"],
              [0.37777777, "rgb(83, 216, 134)"],
              [0.50000000, "rgb(0, 195, 144)"],
              [0.66666666, "rgb(2, 176, 180)"],
              [0.77777777, "rgb(57, 121, 176)"],
              [0.90000000, "rgb(85, 88, 152)"],
              [1.00000000, "rgb(105, 51, 125)"]]

pal_base = ['#98062d', '#b67d74', '#a6b975', '#315c45', '#395e77', '#91bcdd', '#a662a8']


def scroll_js(top):
    return '''<script>
                var body = window.parent.document.querySelector(".main");
                body.scrollTo({top: ''' + str(top) + ''', behavior: 'smooth'});
            </script>'''


def scroll_js_to_line(line):
    return '''<script>
                console.log("HEY")
                var body = window.parent.document.querySelector(".main");
                var line = window.parent.document.getElementById("line''' + str(line) + '''");
                body.scrollTo({top: 100, behavior: 'smooth'});
                console.log("''' + str(line) + '''")
                if (line != null) {
                    let y = line.offsetTop;
                    console.log("HEY")
                    console.log(body)
                    console.log(y)
                    body.scrollTo({top: y, behavior: 'smooth'});
                }
            </script>'''


def style_js():
    return '''<script>hljs.highlightAll();</script>'''


def regular_polygon_coord(center, radius, n):
    return [[center[0] + radius * math.sin((2 * math.pi / n) * i),
            center[1] + radius * math.cos((2 * math.pi / n) * i)] for i in range(n)]


def rectangle_coord(height, width, n):
    vertices_per_side = n / 4
    result = []
    side_index = 0
    x_dim = 0
    y_dim = 0
    while (side_index < 4):
        side_count = 0
        delta_x = 0
        delta_y = 0
        if (side_index == 0):
            delta_x = width / vertices_per_side
        if (side_index == 1):
            delta_y = height / vertices_per_side
        if (side_index == 2):
            delta_x = -width / vertices_per_side
        if (side_index == 3):
            delta_y = -height / vertices_per_side
        while (side_count < vertices_per_side):
            side_count += 1
            x_dim += delta_x
            y_dim += delta_y
            result.append([x_dim, y_dim])
        side_index += 1
    return result



def check_src_code_folder():
    global home_folder, existing_src_code_files, ops_by_line_info
    required_files: set(str) = {i.relative_file_path() for i in CodeLineInfoRow.table()}

    def check_if_exists(fp: str):
        global home_folder
        mod_path = os.path.join(home_folder, fp)
        return os.path.exists(mod_path)

    path_status = {i: check_if_exists(i) for i in required_files}
    for key, value in path_status.items():
        if value is False:
            st.write(Exception(f"The source code file for {os.path.join(home_folder,key)} cannot be found"))
    st.write(CodeLineInfoRow.inferred_home_dir)
    st.json([
            (i.dir_path)
            for i in CodeLineInfoRow.table()
            if len(i.file) > 0 or len(i.dir_path) > 0
        ])

    existing_src_code_files = {
        i.combined_filepath() : os.path.join(home_folder, i.relative_file_path())
        for i in CodeLineInfoRow.table()
        if len(i.file) > 0 and path_status.get(i.relative_file_path()) is True
        }
           



def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=min(0.85, s * 1.75))


def get_object_view_data(gpu_filter=None, allowed_ops=[]):
    global ops_by_line_info
    if gpu_filter == None:
        gpu_filter = [i for i in range(gpu_num)]

    st.json(gpu_filter)

    def check_filter(hex_addr, filtered_ops: List[OpInfoRow]) -> List[OpInfoRow]:
        filtered_addrs = [i for i in filtered_ops if i.addr == hex_addr]
        return filtered_addrs


    cols_rows_name = ['GPU%d' % i for i in range(gpu_num)]
    object_view = []
    for dev in range(gpu_num):
        object_view.append({})

    all_ops: List[OpInfoRow] = OpInfoRow.table()
    print("OWD 1")
    for source_dev_id in range(gpu_num):
        owned_objs: List[SnoopieObject] = [
            i for i in SnoopieObject.all_objects.values()
            if source_dev_id in {j.dev_id for j in i.addres_ranges.keys()}
            ]
        if owned_objs == None:
            continue
        for owned_obj in owned_objs:
            related_accesses: List[OpInfoRow] = [
                i for i in owned_obj.ops 
                if i.running_dev_id in gpu_filter 
                and i.mem_dev_id == source_dev_id
                and i.op_code in allowed_ops ]
    
            if len(related_accesses) == 0:
                break

            related_dict: Dict[str, List[OpInfoRow]] = {}
            for i in related_accesses:
                if i.addr not in related_dict:
                    related_dict[i.addr] = []
                related_dict[i.addr].append(i)

            # get object informations here
            id_info, name_info = related_accesses[0].get_obj_info()
            offset = id_info.offset
            obj = [[]]
            object_map = [[]]
            obj_size = int(id_info.size / ADDR_DIST)
            step = ADDR_DIST
            if obj_size > GRAPH_SIZE_LIMIT:
                step *= int(obj_size / GRAPH_SIZE_LIMIT)
            cols = int(math.sqrt(obj_size)) * 4
            print(related_dict)
            for i in range(0, int(id_info.size), step):
                hex_addr = str.format('0x{:016x}', int(offset, 16) + i)
                # st.write(f"{hex_addr}, {hex_addr in related_dict}")
                if hex_addr not in related_dict: continue
                filtered_addrs: List[OpInfoRow] = related_dict[hex_addr]
                if len(filtered_addrs) > 0:
                    dict_details = {"by_dev": {}, "by_op": {}, "by_line": {}, "combined": {}}
                    for j in filtered_addrs:
                        dev = f"GPU{j.running_dev_id}"
                        op = j.op_code
                        line_info = j.get_line_info()
                        tmp = dict_details["by_dev"].get(dev, 0)
                        dict_details["by_dev"][dev] = tmp + 1
                        tmp = dict_details["by_op"].get(op, 0)
                        dict_details["by_op"][op] = tmp + 1

                        # tmp = dict_details["by_line"].get(line_info, 0)
                        # dict_details["by_line"][line_info] = tmp + 1

                        tmp = line_info.codeline_info.code_line_index
                        if tmp not in ops_by_line_info["by_line_index"]:
                            ops_by_line_info["by_line_index"][tmp] = 0
                        ops_by_line_info["by_line_index"][tmp] += 1

                        tmp = line_info.codeline_info.combined_filepath()
                        if tmp not in ops_by_line_info["by_file"]:
                            ops_by_line_info["by_file"][tmp] = 0
                        ops_by_line_info["by_file"][tmp] += 1

                        tmp2 = line_info.codeline_info.code_linenum
                        if tmp not in ops_by_line_info["by_file_linenum"]:
                            ops_by_line_info["by_file_linenum"][tmp] = {}
                        if tmp2 not in ops_by_line_info["by_file_linenum"][tmp]:
                            ops_by_line_info["by_file_linenum"][tmp][tmp2] = []
                        ops_by_line_info["by_file_linenum"][tmp][tmp2].append(line_info)

                        ops_by_line_info["total"] += 1

                        combined_info = f"{dev}, {op}, {str(line_info)}"
                        tmp = dict_details["combined"].get(combined_info, 0)
                        dict_details["combined"][combined_info] = tmp + 1
                    html_details = "<br>"
                    for key, value in dict_details["combined"].items():
                        html_details += " " * 8 + str(key) + ": " + str(value) + "<br>"
                    for _key, value in dict_details["by_line"].items():
                        
                        pass
                    obj[-1].append((hex_addr, html_details))
                    object_map[-1].append(len(filtered_addrs))
                else:
                    obj[-1].append((hex_addr, ""))
                    object_map[-1].append(0)
            object_view[source_dev_id][owned_obj] = [obj, object_map]
    # st.json(object_view)
    return object_view


def main():
    global gpu_num, ops, chosen_line, ops_to_display, data_by_line, reverse_table_lineinfo, file_to_vis

    top_cols = st.columns([5, 5])

    with top_cols[0]:
        st.radio(
            "Communication units",
            ["Data transfers", "Bytes"],
            key="units",
            horizontal=True
        )
    with top_cols[1]:
        print("ops:", ops)
        ops_to_display = st.multiselect("Operations to display", options=ops, default=ops)


    nodes = []
    edges = []
    max_size = 40.0
    max_width = 10.0
    sizes = []
    pal_base = ['#98062d', '#b67d74', '#a6b975', '#315c45', '#395e77', '#91bcdd', '#a662a8']
    # https://colorkit.co/palette/413344-614c65-806485-936397-a662a8-664972-463c57-6e8da9-
    # 91bcdd-567d99-395e77-305662-264d4d-315c45-8a9a65-b6b975-b65d54-b60033-98062d-800022/
    pal_base = sns.blend_palette(pal_base, n_colors=8)
    pal = sns.color_palette([scale_lightness(color, 1.2) for color in pal_base]).as_hex()
    pal2 = sns.color_palette([scale_lightness(color, 1.0) for color in pal_base]).as_hex()

    # graph_width = 1200
    graph_width = 600
    # graph_height = 800
    graph_height = 400
    font_size = int(graph_height / 22)
    margin = 30

    graph_shape = GRAPH_SHAPES["polygon"]
    positions = []

    if (graph_shape == GRAPH_SHAPES["polygon"]):
        positions = regular_polygon_coord([int(graph_width / 2), int(graph_height / 2)],
                                          int(min(graph_width, graph_height) / 2) - margin, gpu_num)
    elif (graph_shape == GRAPH_SHAPES["square"]):
        positions = rectangle_coord(graph_height, graph_height, gpu_num)
    elif (graph_shape == GRAPH_SHAPES["rectangle"]):
        positions = rectangle_coord(graph_height, graph_width, gpu_num)

    # reduce whitespace on top
    st.markdown("""
        <style>
               .block-container  {
                    margin-top: """ + str(margin * 1.5) + """px;
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

    if (graph_width > graph_height):
        ratio = (graph_width) / (graph_height)
        if (graph_shape == GRAPH_SHAPES["polygon"]):
            for pos in positions:
                pos[0] += ratio * (pos[0] - graph_width / 2) / 2
                pos[0] = max(margin, pos[0])
                pos[0] = min(graph_width - margin, pos[0])


    cols = math.ceil(math.sqrt(gpu_num))
    rows = int(gpu_num / cols)
    # delta_pos = [int((graph_width-2*margin)/max(1, cols-1)), int((graph_height-2*margin)/max(1, rows-1))]

    label_bytes = ""

    if (st.session_state.units == "Bytes"):
        label_bytes = "bytes"

    all_devices = [i for i in range(gpu_num)]
    for i in range(gpu_num):
        label = "GPU" + str(i)

        filtered_ops = OpInfoRow.filter_by_device_and_ops(ops_to_display, [i], all_devices)
        accesses = OpInfoRow.get_total_accesses(filtered_ops, label_bytes == "bytes")
        sizes.append(accesses)

    norm_ratio = max(sizes) / max_size
    if norm_ratio == 0:
        norm_ratio = 1

    for i in range(gpu_num):
        label = "GPU" + str(i)
        nodes.append(Node(id=i, color=pal[i % len(pal)], title=str(sizes[i]), label=label,
                          font={"face": "verdana", "size": font_size, "color": "#000000",
                                "vadjust": -int(sizes[i] / norm_ratio) - font_size},
                          size=int(sizes[i] / norm_ratio), x=positions[i][0], y=positions[i][1]))

    widths = []
    max_val = 0
    for i in range(gpu_num):
        src_label = "GPU" + str(i)
        widths.append([])
        for j in range(gpu_num):
            target_label = "GPU" + str(j)
            pair = src_label + "-" + target_label
            filtered_ops = OpInfoRow.filter_by_device_and_ops(ops_to_display, [i], [j])
            filtered_accesses = OpInfoRow.get_total_accesses(filtered_ops, label_bytes == "bytes")
            width = filtered_accesses * sampling_period
            widths[i].append(width)
            if width > max_val:
                max_val = width
    norm_ratio = max(max(widths)) / max_width
    if norm_ratio == 0:
        norm_ratio = 1
    drawn = [[False] * gpu_num] * gpu_num

    for i in range(gpu_num):
        src_label = "GPU" + str(i)
        for j in range(gpu_num):
            # if (widths[i][j] > 0):
            type = "curvedCW"
            if (drawn[j][i]):
                type = "curvedCCW"
            drawn[i][j] = True
            if i == j:
                continue
            edges.append(Edge(source=i, target=j, hidden=widths[i][j] == 0, color=pal2[i % len(pal2)], type=type,
                              smooth={"enabled": True, "type": type, "roundness": 0.15},
                              font={"face": "verdana", "size": font_size, "color": "#000000"},
                              label=str(widths[i][j]), width=int(widths[i][j] / norm_ratio)))

    config = Config(width=graph_width,
                    height=graph_height,
                    directed=True,
                    physics=False,
                    nodeHighlightBehavior=False,
                    staticGraph=True,
                    highlightColor="#F7A7A6",  # or "blue"
                    # **kwargs
                    )

    cols = st.columns([7, 4])
    # cols = st.columns([100, 1])

    chosen_id_graph = None
    with cols[0]:
        chosen_id_graph = str(agraph(nodes=nodes,
                                     edges=edges,
                                     config=config))

    cols_rows_name = ['GPU%d' % i for i in range(gpu_num)]

    color_range = [0, max_val]

    # modify_cell((H0: 0)) 2  != (H1: 0) 0
    chosen_point = None
    with cols[1]:
        color_title = "Data transfer<br>count"
        if (label_bytes != ""):
            color_title = "Transferred<br>bytes"
        df = pd.DataFrame(widths, columns=cols_rows_name, index=cols_rows_name).astype('int')
        fig = px.imshow(df, color_continuous_scale=colorscale,
                        labels=dict(x="Owner", y="Issued by", color=color_title))
        fig.update_traces(colorbar=dict(lenmode='fraction', len=0.5, thickness=10))
        fig.update_layout(
            font_family="Open Sans, sans-serif",
            # font_color="#fafafa",
            # paper_bgcolor="#0e1117",
            # paper_bgcolor="#e6e6e6",
            font_color="#1a1a1a",
            paper_bgcolor="#ffffff",
        )
        fig.update_yaxes(title_standoff=10)

        fig.layout.width = graph_width * (4.0 / 7.0)
        fig.layout.height = graph_height
        chosen_point = plotly_events(fig)
        if len(chosen_point) > 0:
            chosen_point = chosen_point[0]['pointNumber']
        else:
            chosen_point = None

    if chosen_point != None:
        chosen_id_graph = chosen_point[0]

    chosen_id_tab = stx.tab_bar(data=[
        stx.TabBarItemData(id=str(i), title="GPU" + str(i), description="") for i in range(gpu_num)], default=0)

    # object_view = []
    # for dev in range(gpu_num):
    #     object_view.append({})

    selected_rows = None

    print("TEST 3 ASDASD")
    # object_view = get_object_view_data(gpu_filter=None, allowed_ops=ops_to_display)
    # print(object_view)
    print("TEST 2 ASDASD")
    for i in range(gpu_num):
        if chosen_id_tab == str(i):
            st.markdown(f"### Objects owned by **<span style='color:{pal[i]}'>GPU{i}</span>**", unsafe_allow_html=True)

            objects_owned_cols = st.columns([4, 7])
            other_gpus_selector = [s for s in cols_rows_name if s != f"GPU{i}"]
            with objects_owned_cols[0]:
                filter_chooser = st.multiselect("Filter accesses by GPU", other_gpus_selector, default=other_gpus_selector)

            object_view = get_object_view_data([int(i[3:]) for i in filter_chooser], allowed_ops=ops_to_display)
            if len(object_view[i]) <= 0:
                st.write("No accesses was made with the current filters")
            else:
                is_all_zeros = True
                for key in object_view[i].keys():
                    obj_data = object_view[i][key]
                    for arr in obj_data[1]:
                        for elem in arr:
                            if elem != 0:
                                is_all_zeros = False
                            if not is_all_zeros:
                                break
                        if not is_all_zeros:
                            break
                    if not is_all_zeros:
                        break
                print("all zeros:", is_all_zeros)

                obj_fig = make_subplots(rows=len(object_view[i]),
                                        shared_xaxes=True, cols=1, vertical_spacing=0.05)
                index = 1
                obj_names = []
                for key in object_view[i].keys():
                    obj_data = object_view[i][key]
                    print(obj_data)
                    obj_names.append(key)
                    obj_fig.add_trace(go.Heatmap(z=obj_data[1], coloraxis="coloraxis", name=str(key),
                                                customdata=obj_data[0],
                                                # hovertemplate="Object=%s<br>Offset=%%{x}<br>Instructions=%%{z}<br> \
                                                #  Custom=%{customdata[0]}<extra></extra>"% key), row=index, col=1)
                                                hovertemplate="<br>".join([
                                                    "Offset: %{x}",
                                                    "Instructions: %{z}",
                                                    "Address: %{customdata[0]}",
                                                    "Details: %{customdata[1]}"
                                                ])), row=index, col=1)
                    if index > 1:
                        # obj_fig.update_coloraxes(showscale=False, row=index, col=1)
                        obj_fig['layout']['yaxis' + str(index)].update(dict(tickvals=[0], ticktext=[str(key) + '']))
                    else:
                        obj_fig['layout']['yaxis' + str(index)].update(dict(tickvals=[0], ticktext=[str(key) + '']))

                    obj_fig.update_layout(coloraxis_colorbar=dict(title="Data transfer<br>count"))
                    index += 1

                if is_all_zeros:
                    obj_fig.update_layout(
                        {
                            "coloraxis_cmin": 0,
                            "coloraxis_cmax": 10,
                        }
                    )

                obj_fig.update_layout(coloraxis=dict(colorscale=colorscale), showlegend=False)
                obj_fig.update_layout(
                    margin=dict(l=120, r=120, t=20, b=60),
                    font_family="Open Sans, sans-serif",
                    # font_color="#fafafa",
                    # paper_bgcolor="#0e1117",
                    # paper_bgcolor="#e6e6e6"
                    font_color="#1a1a1a",
                    paper_bgcolor="#ffffff",
                    width=graph_width * (3 / 2)
                )
                obj_fig['layout']['xaxis' + str(index - 1)].update(dict(title="Offset", title_standoff=8))

                chosen_addr = plotly_events(obj_fig)

                st.markdown("""---""")

                if 'ydim' not in st.session_state:
                    st.session_state['ydim'] = 1

                def calc_dim_y():
                    st.session_state.ydim = int(len(object_view[i][obj_option][1][0]) / st.session_state.xdim)

                obj_2d_cols = st.columns([4, 2, 1])
                obj_option = None

                with obj_2d_cols[0]:
                    obj_option = st.selectbox('Choose an object to view in 2D', obj_names)
                    st.write('You selected:', str(obj_option))

                if 'xdim' not in st.session_state:
                    st.session_state['xdim'] = len(object_view[i][obj_option][1][0])

                with obj_2d_cols[1]:
                    xdim = st.number_input('X-dimension', min_value=1, on_change=calc_dim_y(), key='xdim')
                    dim_cols = st.columns([1, 1])
                    with dim_cols[0]:
                        st.write('X-dimension is ', xdim)
                    with dim_cols[1]:
                        st.write('Y-dimension is ', st.session_state.ydim)
                with obj_2d_cols[2]:
                    st.write(' ')
                    st.write(' ')
                    obj_2d_button = st.button('Show in 2D')
                if obj_2d_button:
                    if obj_option is not None:
                        obj_data = object_view[i][obj_option]
                        reshaped_data = []
                        reshaped_details = []
                        ind_x = 0
                        ind_y = 0
                        # reshape both the data and the additional information to be shown on hover
                        for item in obj_data[1][0]:
                            if (ind_x == 0):
                                reshaped_data.append([])
                                reshaped_details.append(obj_data[0][0][ind_y * xdim:(ind_y * xdim) + xdim])

                            reshaped_data[-1].append(item)
                            ind_x += 1
                            if (ind_x == xdim):
                                ind_x = 0
                                ind_y += 1

                        fig = go.Figure(data=go.Heatmap(z=reshaped_data, coloraxis="coloraxis", name=str(key),
                                        customdata=reshaped_details,
                                        xgap=2, ygap=2,
                                        # hovertemplate="Object=%s<br>Offset=%%{x}<br>Instructions=%%{z}<br> \
                                                        #  Custom=%{customdata[0]}<extra></extra>"% key), row=index, col=1)
                                                        hovertemplate="<br>".join(["X-offset: %{x}", "Y-offset: %{y}", "Instructions: %{z}",
                                                                                "Address: %{customdata[0]}",
                                                                                "Details: %{customdata[1]}"
                                                                                ])))
                        fig.update_layout(coloraxis_colorbar=dict(title="Data transfer<br>count"))
                        index += 1

                        fig.update_layout(coloraxis=dict(colorscale=colorscale), showlegend=False)
                        fig.update_layout(
                            margin=dict(l=120, r=120, t=20, b=60),
                            font_family="Open Sans, sans-serif",
                            # font_color="#fafafa",
                            # paper_bgcolor="#0e1117",
                            # plot_bgcolor="#0e1117",
                            font_color="#1a1a1a",
                            # paper_bgcolor="#e6e6e6",
                            # plot_bgcolor="#e6e6e6",
                            paper_bgcolor="#ffffff",
                            plot_bgcolor="#ffffff",
                            yaxis=dict(autorange="reversed", zeroline=False),
                            xaxis=dict(visible=False),
                            width=graph_width
                        )
                        chosen_cell = plotly_events(fig)

            st.markdown("""---""")
            st.markdown(f"### Communication issued by **<span style='color:{pal[i]}'>GPU{i}</span>**", unsafe_allow_html=True)

            cols = st.columns([1 for i in range(3)])
            other_gpus = [i for i in range(gpu_num)]
            other_gpus.remove(i)
            table_objs = []
            table_instr = []
            table_lines = []
            by_var_name_onclick: List[LineInfo] = []
            to_filter = OpInfoRow.filter_by_device_and_ops(allowed_ops=ops_to_display, allowed_mem_devs=[i for i in range(gpu_num)], allowed_running_devs=[i])
            table_instr.append(("total", len(to_filter), "-"))
            table_objs.append(("-","total", len(to_filter), "-"))
            for peer_gpu in other_gpus:
                ops_accessed: List[OpInfoRow] = [op for op in to_filter if op.mem_dev_id == peer_gpu]

                by_op_type = {}
                by_var_name: Dict[LineInfo, int] = {}
                by_lines = {}
                for op in ops_accessed:
                    if op.op_code not in by_op_type.keys():
                        by_op_type[op.op_code] = 0
                    by_op_type[op.op_code] = by_op_type[op.op_code] + 1
                    op_id_info, op_name_info = op.get_obj_info()
                    line_info: LineInfo = op.get_line_info(op_id_info, op_name_info)
                    if line_info not in by_var_name.keys():
                        by_var_name[line_info] = 0
                        by_var_name_onclick.append(line_info)
                    by_var_name[line_info] = by_var_name[line_info] + 1
                    line_display_str = ""
                    line_display_str = f"{line_info.codeline_info.file}: line{line_info.codeline_info.code_linenum}"
                    if line_display_str not in by_lines:
                        by_lines[line_display_str] = 0
                    by_lines[line_display_str] = by_lines[line_display_str] + 1
                    reverse_table_lineinfo[line_display_str] = line_info

                for key, value in by_op_type.items():
                    table_instr.append((key, value, peer_gpu))
                for key, value in by_var_name.items():
                    table_objs.append((key.obj_id_info.obj_id,key.obj_name_info.var_name, value, peer_gpu))
                for key, value in by_lines.items():
                    table_lines.append((key, value, peer_gpu))

            ag_custom_css = {
                "#gridToolBar": {
                    "padding-bottom": "0px !important",
                }
            }

            if "object_table_selected" not in st.session_state:
                st.session_state.object_table_selected = None

            if len(table_objs) > 0:
                cols[0].markdown("##### Objects", unsafe_allow_html=True)
                df = pd.DataFrame.from_dict(table_objs)
                df.columns = ['object id', 'var name', 'count', 'destination']
                table_height = min(MIN_TABLE_HEIGHT + len(df) * (ROW_HEIGHT), MAX_TABLE_HEIGHT)
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_selection('single', use_checkbox=False, pre_selected_rows=None)
                gridOptions = gb.build()
                with cols[0]:
                    grid_response = AgGrid(df, gridOptions, height=table_height, fit_columns_on_grid_load=True, custom_css=ag_custom_css,
                           columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                           update_mode=GridUpdateMode.SELECTION_CHANGED 
                           )
                    selected_rows = grid_response['selected_rows']
                    print("RESPONSEAAAAAAAAAAAAAAA",grid_response)
                    if (len(selected_rows) > 0) and selected_rows[0]["_selectedRowNodeInfo"]["nodeRowIndex"] > 0:
                        st.session_state.object_table_selected = selected_rows[0]["_selectedRowNodeInfo"]["nodeRowIndex"] - 1
                cols[1].markdown(f"##### Instructions", unsafe_allow_html=True)
                df = pd.DataFrame.from_dict(table_instr)
                df.columns = ['instruction', 'count', 'destination']
                table_height = min(MIN_TABLE_HEIGHT + len(df) * (ROW_HEIGHT), MAX_TABLE_HEIGHT)
                with cols[1]:
                    AgGrid(df, height=table_height, fit_columns_on_grid_load=True, custom_css=ag_custom_css,
                           columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)

                cols[2].markdown(f"##### Code lines", unsafe_allow_html=True)
                if len(table_lines) == 0:
                    table_lines = [("-","-","-")]
                df = pd.DataFrame.from_dict(table_lines)
                df.columns = ['code line', 'count', 'destination']
                table_height = min(MIN_TABLE_HEIGHT + len(df) * (ROW_HEIGHT), MAX_TABLE_HEIGHT)
                with cols[2]:
                    gb = GridOptionsBuilder.from_dataframe(df)
                    gb.configure_selection('single', use_checkbox=False, pre_selected_rows=None)
                    gridOptions = gb.build()
                    grid_response = AgGrid(df, gridOptions, height=table_height, fit_columns_on_grid_load=True,
                                           update_mode=GridUpdateMode.SELECTION_CHANGED, custom_css=ag_custom_css,
                                           columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
                    selected_rows = grid_response['selected_rows']
                    if (len(selected_rows) > 0):
                        if (selected_rows[0]['code line'] != 'total'):
                            chosen_file, chosen_line = selected_rows[0]['code line'].split(':')
                            chosen_line = int(chosen_line.strip()[4:])
                            chosen_file = os.path.join(CodeLineInfoRow.inferred_home_dir,  chosen_file.strip())
                            tkey = 'table_select' + str(peer_gpu)
                            st.session_state[tkey] = chosen_line
                            file_to_vis = chosen_file
                            if (tkey not in st.session_state or st.session_state[tkey] != chosen_line):
                                show_sidebar()
                            # show_sidebar(int(selected_rows['code line']))
                            # .scrollTop = ''' + str(graph_height + i/100) + ''';
                components.html(scroll_js(graph_height + margin + i * 0.0001), height=0)
                if st.session_state.object_table_selected != None:
                    if not (len(by_var_name_onclick) <= st.session_state.object_table_selected or st.session_state.object_table_selected < 0):
                        tempop: LineInfo = by_var_name_onclick[st.session_state.object_table_selected]
                        st.markdown(f"##### Selected Object: {tempop.obj_id_info.obj_id}-{tempop.obj_name_info.var_name}", unsafe_allow_html=True)
                        tempdict = [
                            (i.file_name, i.func_name, i.line_no) for i in tempop.call_stack
                        ]
                        df = pd.DataFrame.from_dict(tempdict)
                        df.columns = [ 'file name', 'function name', 'line']
                        table_height = min(MIN_TABLE_HEIGHT + len(df) * (ROW_HEIGHT), MAX_TABLE_HEIGHT)
                        AgGrid(df, height=table_height, fit_columns_on_grid_load=True, custom_css=ag_custom_css,
                            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                            )

                break
            else:
                st.write("This gpu issued no communication")
                break


def show_sidebar():
    global chosen_line, file_to_vis, ops_by_line_info
    # print("CHOSEN LINE " + str(chosen_line))
    print("XDE", file_to_vis not in ops_by_line_info["by_file_linenum"])
    if file_to_vis not in ops_by_line_info["by_file_linenum"]: return
    linenum = chosen_line
    if (linenum in ops_by_line_info["by_file_linenum"][file_to_vis]):
        with st.sidebar:
            linedata: List[LineInfo] = ops_by_line_info["by_file_linenum"][file_to_vis][chosen_line]
            st.markdown(
                f'''
                <style>
                    .css-1544g2n {{
                        padding-top: 0px;
                        margin-top: 0px;
                        margin-right: {-6}rem;
                    }}
                    .css-ysnqb2 {{
                        margin-top: {0}rem;
                        padding-top: {0}rem;
                        padding-right: {-2}rem;
                        margin-left: -60px;
                        padding-bottom: {0}rem;
                    }}
                </style>
                ''',
                unsafe_allow_html=True)
            st.markdown("# Line " + str(linenum))
            st.markdown(
                """
                <style>
                :root {
                    background-color: #f5f5f5;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("""<head><script>hljs.highlightAll();</script></head><body>
                            <code class='hljs language-c'>""" + src_lines[file_to_vis][linenum - 1] + """</code></body>""",
                        unsafe_allow_html=True,)
            st.markdown("### Total transfers: " + str(len(linedata)))
            st.markdown("#### Objects updated: ")
            tmpcounter: Dict[LineInfo, int] = {}
            for i in linedata:
                tmpcounter[i] = 0
            for i in linedata:
                tmpcounter[i] += 1
            for i in tmpcounter.keys():
                var_name = i.obj_name_info.var_name
                call_stack_str = ""
                for fnc_call in i.call_stack:
                    call_stack_str += fnc_call.func_name + " -> "
                if var_name is None or var_name == "": var_name = "[unnamed object]"
                call_stack_str += var_name
                st.markdown("###### " + call_stack_str + ": " + str(tmpcounter[i]))

            # obj_str = "### Object(s) involved: \\"
            # for obj in linedata.get('objects', set()):
            #     obj_str += str(obj) + ": " + str(linedata[obj]) + ""

            # st.markdown(obj_str)

            cols_rows_name = ['GPU%d' % i for i in range(gpu_num)]

            widths = []

            for i in range(gpu_num):
                src_label = "GPU" + str(i)
                widths.append([])
                for j in range(gpu_num):
                    target_label = "GPU" + str(j)
                    pair_label = src_label + "-" + target_label
                    width = 0.0
                    #TODO we can also do the samething with linedata to get filtered result by gpu
                    filtered_ops = [p for p in OpInfoRow.table()
                                    if p.op_code in ops_to_display
                                    and p.mem_dev_id == j
                                    and p.running_dev_id == i
                                    and (p.get_line_info()).check_correct_line(file_to_vis, linenum)]
                    width = len(filtered_ops)
                    widths[i].append(width)

            df = pd.DataFrame(widths, columns=cols_rows_name, index=cols_rows_name).astype('int')
            fig = px.imshow(df, color_continuous_scale=colorscale,
                            labels=dict(x="Owner", y="Issued by", color="Data transfer<br>count"))
            fig.update_traces(colorbar=dict(lenmode='fraction', len=0.5, thickness=10))
            fig.update_layout(
                font_family="Open Sans, sans-serif",
                # font_color="#fafafa",
                # paper_bgcolor="#0e1117",
                # paper_bgcolor="#e6e6e6",
                font_color="#1a1a1a",
                paper_bgcolor="#f5f5f5",
            )
            fig.update_yaxes(title_standoff=10)

            fig.layout.width = 300
            fig.layout.height = 300
            chosen_point = plotly_events(fig)
            # print(chosen_point)
            if len(chosen_point) > 0:
                chosen_point = chosen_point[0]['pointNumber']
            else:
                chosen_point = None
            st.session_state.sidebar_state = 'expanded'


def read_code():
    global src_lines, existing_src_code_files

    iterate_set: set(str) = {i.combined_filepath() for i in CodeLineInfoRow.table()}

    for i in iterate_set:
        if i in existing_src_code_files:
            filepath = existing_src_code_files[i]
            if os.path.exists(filepath) and os.path.isfile(filepath):
                with open(filepath, "r") as f:
                    src_lines[i] = f.readlines() 
    
    print("src_lines:", {key: len(value) for key, value in src_lines.items()})

def process_folderpath(folder: str):
    if folder[-1] == "/": folder = folder[:-1]
    return folder

def choose_src_code_file():
    global file_to_vis, existing_src_code_files, home_folder, current_folder, ops_by_line_info, choose_src_code_file_list

    def update_clicked_image(index):
        if index not in choose_src_code_file_list:
            choose_src_code_file_list[index] = 0
        choose_src_code_file_list[index] += 1

    files_to_show = set()
    folders_to_show = set()

    if current_folder == None:
        current_folder = home_folder
    
    # setup files to show
    for filepath in existing_src_code_files.values():
        folder, file = os.path.split(filepath)
        if process_folderpath(folder) == current_folder:
            files_to_show.add(file)

    # setup folders to show
    for filepath in existing_src_code_files.values():
        folder, file = os.path.split(filepath)
        folder = process_folderpath(folder)
        # print(folder, current_folder, home_folder, CodeLineInfoRow.inferred_home_dir)
        if folder == current_folder: continue
        commonpath = os.path.commonpath((folder, current_folder))
        if commonpath == current_folder:
            folder_cut = folder[len(commonpath)+1:]
            if folder_cut.count(os.path.sep) > 0:
                continue
            folders_to_show.add(folder_cut)

    print("fsp", files_to_show, folders_to_show)

    # display them
    # TODO download them so they also work offline
    folder_icon, file_icon = ("https://icons.getbootstrap.com/assets/icons/folder.svg", "https://icons.getbootstrap.com/assets/icons/file-earmark.svg")
    img_size = 32

    temp = sorted(list(folders_to_show))
    if current_folder != home_folder:
        temp.insert(0, "..")
    combined_list = temp  + sorted(list(files_to_show))
    totallen = len(files_to_show) + len(temp)
    folder_len = len(temp)
    item_per_line = 8
    idx = 0
    st.write(f"current folder: {current_folder}")
    while totallen > 0:
        cols = st.columns([1 for i in range(item_per_line)])
        for i in range(item_per_line):
            if idx not in choose_src_code_file_list:
                choose_src_code_file_list[idx] = 0
            totallen -= 1
            if totallen < 0: break
            name = combined_list[idx]
            fake_item = os.path.join(CodeLineInfoRow.inferred_home_dir, current_folder[len(home_folder)+1:],name)
            print(current_folder, name, fake_item)
            with cols[i]:
                if name == "..":
                    # go back
                    clicked = clickable_images([folder_icon],titles=[name], img_style={"width":f"{img_size}px","height":f"{img_size}px"})
                    st.write(name)
                    if clicked > -1:
                        st.session_state.current_folder, _ = os.path.split(current_folder)
                        setup_globals()
                        choose_src_code_file_list = {}
                        st.experimental_rerun()
                elif idx < folder_len:
                    #display folder
                    print("display folder:", name)
                    clicked = clickable_images([folder_icon],titles=[name], img_style={"width":f"{img_size}px","height":f"{img_size}px"})
                    st.write(name)
                    if clicked > -1:
                        st.session_state.current_folder = os.path.join(current_folder, name)
                        setup_globals()
                        choose_src_code_file_list = {}
                        st.experimental_rerun()
                else:
                    #display file
                    print("display file:", name)
                    num_accesses = ops_by_line_info["by_file"]

                    clicked = clickable_images([file_icon],titles=[name], img_style={"width":f"{img_size}px","height":f"{img_size}px"}, div_style={"asd" : f"{choose_src_code_file_list[idx]}"})
                    
                    st.write(name)
                    
                    access_count = 0 if fake_item not in ops_by_line_info['by_file'] else ops_by_line_info['by_file'][fake_item]
                    if ops_by_line_info['total'] == 0: ops_by_line_info['total'] = 1
                    
                    st.write(f"Accesses: {access_count}, %{100 * access_count /ops_by_line_info['total']}")

                    if clicked > -1:
                        file_to_vis = fake_item
                        update_clicked_image(idx)
                        
            idx += 1


def show_code():
    global src_lines, chosen_line, ops_to_display, ops_by_line_info, current_folder, file_to_vis
    content_head = """<head>
        <style>
            .percentage {
                position:absolute;
                top:0;
                left:0;
                z-index:-1;
                height:15px;
            }
        </style>
        <link rel="stylesheet"
            href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">""" + \
        (f"<script>{electron_checker.electron_load_highlight()}</script>" if electron_checker.is_electron else """
        <script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
        """) + """
        <script>hljs.highlightAll();</script>
        </head><body><pre>"""
    content_end = """</pre></body>"""

    max_line_comm = 0
    total_comm = ops_by_line_info['total']
    filtered_comm = {}

    for line_index, linedata in ops_by_line_info["by_file_linenum"][file_to_vis].items():
        ftotal = len(linedata)
        filtered_comm[line_index] = ftotal
        if (ftotal > max_line_comm):
            max_line_comm = ftotal

    if total_comm == 0:
        total_comm = 1  # division by zero
    # https://colorkit.co/palette/413344-614c65-806485-936397-a662a8-664972-463c57-6e8da9-
    # 91bcdd-567d99-395e77-305662-264d4d-315c45-8a9a65-b6b975-b65d54-b60033-98062d-800022/
    pal_base_lines = sns.blend_palette(pal_base[:-3], n_colors=32)
    pal = sns.color_palette([scale_lightness(color, 1.2) for color in pal_base_lines]).as_hex()
    pal.reverse()

    total_lines = len(src_lines[file_to_vis])
    index_chars = len(str(total_lines)) + 1
    line_index = 0
    for line in src_lines[file_to_vis]:
        line_index += 1
        ls = len(line) - len(line.lstrip())
        line = line.replace('<', '&lt')
        line = line.replace('>', '&gt')
        # st.code(str(line_index)+ '.  ' + line)
        # content_head+="""<div style='display:flex;flex-direction:row;><code class='hljs language-c'
        #                 style='width:90%;margin-bottom:0;padding-bottom:0;overflow-x:hidden"""
        comm_percentage = 0.0
        if line_index in ops_by_line_info["by_file_linenum"][file_to_vis]:
            comm_percentage = float(filtered_comm[line_index]) / float(total_comm) * 100
        div_inject = "<div class='percentage' style='background-color:" + pal[int(comm_percentage) % len(pal)] + ";opacity:0.4;width:" + str(comm_percentage) + "%'></div>"

        str_percentage = f"{comm_percentage:2.2f}%"
        if (comm_percentage < 10.0):
            str_percentage = " " + str_percentage
        if (comm_percentage == 0):
            str_percentage = " " * 6
        div_inject += "<span style='position:absolute; bottom:0.2rem;'>" + str_percentage + "</span>"
        content_head += """<div id='d""" + str(line_index) + """'style='position:relative;width=100%'>""" + div_inject + """<a 
                        id='line""" + str(line_index) + """' style='display:inline-block;margin-left:2.5rem'><code class='hljs language-c' style='position:relative;width:100%;
                        background-color:rgb(0,0,0,0);margin-bottom:0;padding-bottom:0;overflow-x:hidden;"""
        if line_index == 1:
            content_head += "'>"
        else:
            content_head += ";margin-top:0;padding-top:0'>"
        content_head += str(line_index) + '.' + ''.join([' ' * (index_chars - len(str(line_index)))]) + line + "</a></code></div>"


    clicked_src = click_detector(content_head + content_end)
    st.markdown(
        """
        <style>
        .css-1o14730, .css-2qelbv {
            background-color: #f5f5f5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if clicked_src:
        chosen_line = int(clicked_src[4:])
        print("chosen line: ", chosen_line)
        ckey = 'code_select'
        if (ckey not in st.session_state or st.session_state[ckey] != chosen_line):
            show_sidebar()
            st.session_state[ckey] = chosen_line



def continue_main():
    global gpu_num, ops, logfile, logfile_name, file_to_vis

    gpu_num, ops = read_data(logfile, logfile_name, (gpu_num, ops))
    setup_globals()

    check_src_code_folder()
    read_code()
    print("BEFORE MAIN")
    # st.json(SnoopieObject.get_display_dict())
    st.json([{
        "unique_obj": i.get_unique_obj(),
        "addr": i.addr,
        "offs": i.obj_offset
    } for i in OpInfoRow.table()
    if i.mem_dev_id == 2 and i.running_dev_id == 0])
    main()

    st.markdown("""---""")
    choose_src_code_file()
    st.markdown("""---""")
    if (file_to_vis is not None
    and file_to_vis in ops_by_line_info["by_file"]
    and file_to_vis in ops_by_line_info["by_file_linenum"]):
        show_code()

        


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    argumentparser.parse()
    setup_globals()
    if not st.session_state.show_filepicker:
        start_newfile = st.button("Profile another file")
        if start_newfile:
            # pickle_filename = "".join(logfile_name.split(".")[:-1]) + ".pkl"
            if "pickle_filename" in st.session_state and os.path.exists(st.session_state.pickle_filename):
                os.remove(st.session_state.pickle_filename)
            reset_globals()
            st.session_state.show_filepicker = True

    if st.session_state.show_filepicker:
        if filepicker_page.filepicker_page() == False:
            continue_main()
    else:
        continue_main()
    # print(clicked_src)
