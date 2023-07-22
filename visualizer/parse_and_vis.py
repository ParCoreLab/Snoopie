import sys
import math
import os
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import pickle 
import extra_streamlit_components as stx
from st_click_detector import click_detector
import seaborn as sns
import pandas as pd
import numpy as np
import colorsys
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import json
import zstandard as zstd
import io


data_by_address = {}
data_by_device = {}
data_by_obj = {}
data_by_line = {}
# addr_obj_map = {}

#####################################
#     CHANGE THESE WHEN NEEDED!     #
gpu_num = 4                         #
src_code_file = "four-gpus.cu"     #
#                                   #
#####################################

keys = []
ops = set()
addrs = set()
ptx_code = []
ptx_code_rev = {}

GRAPH_SIZE_LIMIT = 100000
ADDR_DIST = 4
colorscale=[[0.00000000, "rgb(230, 233, 233)"],
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
    return [[center[0] + radius * math.sin((2*math.pi/n) * i),
            center[1] + radius * math.cos((2*math.pi/n) * i)] for i in range(n)]


def isInt_try(v):
    try:     i = int(v)
    except:  return False
    return True


@st.cache_data
def read_data(file):
    global data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, keys, ops, addrs, ptx_code, ptx_code_rev


    f = open(file, "r")

    if file.endswith(".zst"):
        f = open(file, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        f = io.TextIOWrapper(reader, encoding='utf-8')


    if f is None:
        print("File not found")
        exit(0)

    ops_set = set()
    addrs_set = set()


    # prints all files
    graph_name = ""
    pickle_file = None

    pickle_filename = ''.join(file.split(".")[:-1]) + '.pkl'
    if os.path.isfile(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            pickle_file = pickle.load(f)
            print("Data loaded from " + pickle_filename)

    reading_data = 0
    opkeys = []
    objkeys = []
    counter = 0

    if (pickle_file is None):    
        
        for line in f:
            counter+=1
            if counter%100000 == 0:
                print(counter)
            if reading_data:
                data = {}
                vals = line.strip().split(',')
                # print(vals)
                if len(vals) != len(opkeys):
                    reading_data = False
                    if (line.startswith('offset')):
                        objkeys = line.split(', ')
                        reading_data = True
                    break
                for index in range(len(opkeys)):
                    if isInt_try(vals[index]):
                        data[opkeys[index]] = int(vals[index])
                    else:
                        data[opkeys[index]] = vals[index]

                address = data["addr"]
                obj_name = data["obj_offset"]
                operation = data["op_code"]
                linenum = data["code_linenum"]
                # TODO: Use this when computing the overall bytes transfered
                # mem_range = data["mem_range"]

                addrs.add(address)

                ops.add(operation)

                device = "GPU"+str(data["running_dev_id"])
                owner = "GPU"+str(data["mem_dev_id"])
                pair = device+"-"+owner

                # print(operation)
                # print(device)
                # print(owner)
                # print(address)
                # print(obj_name)

                data_by_address[address] = data_by_address.get(address, {})
                temp_data = data_by_address[address]
                temp_data['total'] = temp_data.get('total', 0) + 1
                temp_data[operation] = temp_data.get(operation, 0) + 1
                temp_data[device] = temp_data.get(device, 0) + 1
                temp_data['line_' + str(linenum)] = temp_data.get('line_' + str(linenum), 0) + 1
                temp_lines = temp_data.get('lines', set())
                temp_lines.add('line_' + str(linenum))
                temp_data['lines'] = temp_lines

                data_by_device[pair] = data_by_device.get(pair, {})
                temp_data = data_by_device[pair]
                temp_data['total'] = temp_data.get('total', 0) + 1
                temp_data[operation] = temp_data.get(operation, 0) + 1
                temp_data[obj_name] = temp_data.get(obj_name, 0) + 1
                temp_data['line_' + str(linenum)] = temp_data.get('line_' + str(linenum), 0) + 1
                temp_lines = temp_data.get('lines', set())
                temp_lines.add('line_' + str(linenum))
                temp_data['lines'] = temp_lines
                
                data_by_device[device] = data_by_device.get(device, {})
                temp_data = data_by_device[device]
                temp_data['total'] = temp_data.get('total', 0) + 1
                temp_data[operation] = temp_data.get(operation, 0) + 1
                temp_data[owner] = temp_data.get(owner, 0) + 1
                temp_data[obj_name] = temp_data.get(obj_name, 0) + 1

                data_by_line[linenum] = data_by_line.get(linenum, {})
                temp_data = data_by_line[linenum]
                temp_data['total'] = temp_data.get('total', 0) + 1
                temp_data[pair] = temp_data.get(pair, 0) + 1
                temp_data[operation] = temp_data.get(operation, 0) + 1
                temp_data[address] = temp_data.get(address, 0) + 1
                temp_data[obj_name] = temp_data.get(obj_name, 0) + 1
                temp_objects = temp_data.get('objects', set())
                temp_objects.add(obj_name)
            else:
                if (line.startswith("filename=")):
                    print("This is a graph file!")
                    splt_info = line.strip().split(',')
                    graph_name = splt_info[0].split('/')[-1]
                    gpu_num = int(splt_info[1][-1])
                # op_code, addr, thread_indx, running_dev_id, mem_dev_id, code_linenum, code_line_index, code_line_estimated_status, obj_offset
                elif (line.startswith('op_code')):
                    opkeys = line.strip().split(', ')
                    reading_data = True
                # elif (line.startswith('offset')):
                #     objkeys = line.split(', ')
                #     reading_data = 2
                # elif (line.startswith('global_index')):
                #     splt_line = line.split(', ')
                #     instr = ''.join(splt_line[-3].split[' '][1:])
                #     line_num = int(splt_line[-2].split[' '][1])
                #     if 'estimated' in splt_line[-1]:
                #         line_num = -line_num
                #     ptx_code.append(instr, line_num)

        for line in f:
            if reading_data:
                data = {}
                vals = line.strip().split(',')
                if len(vals) != len(objkeys):
                    reading_data = False
                    continue
                for index in range (len(objkeys)):
                    if isInt_try(vals[index]):
                        data[objkeys[index]] = int(vals[index])
                    else:
                        data[objkeys[index]] = vals[index]
                data_by_obj[data[objkeys[0]]]=data
                # for i in range(data['size']):
                #     hex_addr = str.format('0x{:016x}', int_off+(i*4))
                #     addr_obj_map[hex_addr] = hex_off
            else:
                # offset, size, device_id, var_name, filename, alloc_line_num
                if (line.startswith('offset')):
                    objkeys = line.split(', ')
                    reading_data = True
                elif (line.startswith('global_index')):
                    splt_line = line.split('sass_instruction: ')[1].split(';')
                    instr = splt_line[0] + ';'
                    rh = splt_line[1][2:]
                    line_num = int(rh[10:rh.index(',')])
                    if 'estimated' in splt_line[-1]:
                        line_num = -line_num
                    ptx_code.append([instr, line_num])
                    ptx_code_rev[line_num] = ptx_code_rev.get(line_num, list())
                    ptx_code_rev[line_num].append(instr)

        print("Reading complete")

        # print(graph_name, end=',')
        # print(gpu_num)
        # for key, value in sorted(data_by_device.items(), key=lambda x: x[0]): 
        #     print("{} : {}".format(key, value))
        # print(keys)
        # print(ops)
        # print("ADDRESS")
        # print(data_by_address[list(data_by_address.keys())[0]])
        # print("\n\nDEVICE")
        # print(data_by_device[list(data_by_device.keys())[0]])
        # print("\n\nOBJECT")
        # print(data_by_obj[list(data_by_obj.keys())[0]])
        # for elem in list(data_by_obj.keys()):
        #     print(data_by_obj[elem]['var_name'])

        all_data = [data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, keys, ops, addrs, ptx_code, ptx_code_rev]
        
        with open(pickle_filename, 'wb') as pf:
            pickle.dump(all_data, pf)
            print("Data saved to " + pickle_filename)
    else:
        data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, keys, ops, addrs, ptx_code, ptx_code_rev = pickle_file

    return data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, keys, ops, addrs, ptx_code, ptx_code_rev


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = min(0.85, s * 1.75) )


def main():
    global data_by_address, data_by_device, gpu_num, keys, ops, addrs
    # date = st.radio("Pick a metric", keys)
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

    graph_width = 600
    graph_height = 400
    font_size = int(graph_height/22)
    margin = 20
    positions = regular_polygon_coord([int(graph_width/2), int(graph_height/2)], 
                                      int(min(graph_width, graph_height)/2)-margin, gpu_num)

    # reduce whitespace on top
    st.markdown("""
        <style>
               .block-container  {
                    margin-top: """ + str(margin*1.5) + """px;
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)
    
    if (graph_width > graph_height):
        ratio = (graph_width)/(graph_height)
        for pos in positions:
            pos[0] += ratio*(pos[0]-graph_width/2)/2
            pos[0] = max(margin, pos[0])
            pos[0] = min(graph_width-margin, pos[0])
        

    cols = math.ceil(math.sqrt(gpu_num))
    # print("ATTENTION")
    # print(cols)
    # print(gpu_num)
    rows = int(gpu_num/cols)
    delta_pos = [int((graph_width-2*margin)/max(1, cols-1)), int((graph_height-2*margin)/max(1, rows-1))]

    # print(data_by_device)
    for i in range(gpu_num):
        label = "GPU"+str(i)

        size = 0.0
        if (label in data_by_device):
            size = data_by_device[label]['total']
        sizes.append(size)
    
    norm_ratio = max(sizes)/max_size

    for i in range(gpu_num):
        label = "GPU"+str(i)
        nodes.append(Node(id=i, color=pal[i%len(pal)], title=str(sizes[i]), label=label, 
                          font={"face": "verdana", "size": font_size, "color": "#000000",
                                "vadjust": -int(sizes[i]/norm_ratio)-font_size}, 
                          size=int(sizes[i]/norm_ratio), x=positions[i][0], y=positions[i][1]))
        
    widths = []
    max_val = 0
    for i in range(gpu_num):
        src_label = "GPU"+str(i)
        widths.append([])
        for j in range(gpu_num):
            target_label = "GPU"+str(j)
            width = 0.0
            if (src_label in data_by_device and target_label in data_by_device[src_label]):
                width = data_by_device[src_label][target_label]
            widths[i].append(width)
            if width > max_val:
                max_val = width
    
    norm_ratio = max(max(widths))/max_width
    drawn = [[False] * gpu_num] * gpu_num

    for i in range(gpu_num):
        src_label = "GPU"+str(i)
        for j in range(gpu_num):
            # if (widths[i][j] > 0):
            type = "curvedCW"
            if (drawn[j][i]):
                type = "curvedCCW"
            drawn[i][j] = True
            if i == j:
                continue
            edges.append(Edge(source=i, target=j, hidden=widths[i][j]==0, color=pal2[i%len(pal2)], type=type, 
                              smooth={"enabled": True, "type": type, "roundness": 0.15},
                              font={"face": "verdana", "size": font_size, "color": "#000000"},
                              label=str(widths[i][j]), width=int(widths[i][j]/norm_ratio))) 

    config = Config(width=graph_width,
                    height=graph_height,
                    directed=True, 
                    physics=False, 
                    nodeHighlightBehavior=False, 
                    staticGraph=True,
                    highlightColor="#F7A7A6", # or "blue"
                    # **kwargs
                    )

    cols = st.columns([7, 4])
            
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
            paper_bgcolor="#ffffff",
        )
        fig.update_yaxes(title_standoff = 10)

        fig.layout.width = graph_width*(4.0/7.0)
        fig.layout.height = graph_height
        chosen_point = plotly_events(fig)
        # print(chosen_point)
        if len(chosen_point) > 0:
            chosen_point = chosen_point[0]['pointNumber']
        else:
            chosen_point = None
        # print(chosen_point)

    # print(chosen_id_graph)
    if chosen_point != None:
        chosen_id_graph = chosen_point[0]
    # print(chosen_id_graph)

    chosen_id_tab = stx.tab_bar(data=[
        stx.TabBarItemData(id=str(i), title="GPU"+str(i), description="") for i in range(gpu_num)], default=0)

    object_view = []
    for dev in range(gpu_num):
        object_view.append({})

    for key in data_by_obj.keys():
        offset = int(data_by_obj[key]['offset'], 16)
        obj = [[]]
        object_map = [[]]
        obj_size = int(data_by_obj[key]['size']/ADDR_DIST)
        step = ADDR_DIST
        if obj_size > GRAPH_SIZE_LIMIT:
            step *= int(obj_size/GRAPH_SIZE_LIMIT)
        cols = int(math.sqrt(obj_size))*4
        ycounter = cols
        # set up the data to be displayed in the objectview on hover
        for i in range(0, int(data_by_obj[key]['size']), step):
            # if i >= ycounter:
            #     ycounter += cols
            #     obj.append([])
            #     object_map.append([])
            hex_addr = str.format('0x{:016x}', offset+i)
            if hex_addr in data_by_address.keys():
                dict_details = data_by_address[hex_addr]
                html_details = "<br>"
                for dev in cols_rows_name:
                    if dev in dict_details:
                        html_details += " "*8 + str(dev) + ": " + str(dict_details[dev]) + "<br>"
                for op in ops:
                    if op in dict_details:
                        html_details += " "*8 + str(op) + ": " + str(dict_details[op]) + "<br>"
                for line in dict_details['lines']:
                    html_details += " "*8 + str(line) + ": " + str(dict_details[line]) + "<br>"

                obj[-1].append([hex_addr, html_details])
                object_map[-1].append(dict_details['total'])
            else:
                obj[-1].append([hex_addr, ""])
                object_map[-1].append(0)
        if (data_by_obj[key]['device_id'] >= 0):
            object_view[data_by_obj[key]['device_id']][data_by_obj[key]['var_name']] = [obj, object_map]
   
    for i in range(gpu_num):
        if len(object_view[i]) > 0 and chosen_id_tab == str(i):
            st.markdown(f"### Objects owned by **<span style='color:{pal[i]}'>GPU{i}</span>**", unsafe_allow_html=True)
            
            obj_fig = make_subplots(rows=len(object_view[i]), 
                                    shared_xaxes=True, cols=1, vertical_spacing=0.05)
            index = 1
            obj_names = []
            for key in object_view[i].keys():
                obj_data = object_view[i][key]
                obj_names.append(key)
                obj_fig.add_trace(go.Heatmap(z=obj_data[1], coloraxis="coloraxis", name=key,
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
                    obj_fig['layout']['yaxis' + str(index)].update(dict(tickvals=[0], ticktext=[key + '']))
                else:
                    obj_fig['layout']['yaxis' + str(index)].update(dict(tickvals=[0], ticktext=[key + '']))

                obj_fig.update_layout(coloraxis_colorbar=dict(title="Data transfer<br>count"))               
                index += 1

            obj_fig.update_layout(coloraxis=dict(colorscale=colorscale), showlegend=False)
            obj_fig.update_layout(
                    margin=dict(l=120, r=120, t=20, b=60),
                    font_family="Open Sans, sans-serif",
                    # font_color="#fafafa",
                    # paper_bgcolor="#0e1117",
                    # paper_bgcolor="#e6e6e6"
                    font_color="#1a1a1a",
                    paper_bgcolor="#ffffff",
                    width=graph_width*(3/2)
                )
            obj_fig['layout']['xaxis' + str(index-1)].update(dict(title="Offset", title_standoff=8))
                
            chosen_addr = plotly_events(obj_fig)
            print(chosen_addr)

            st.markdown("""---""")

            if 'ydim' not in st.session_state:
                st.session_state['ydim'] = 1

            def calc_dim_y():
                st.session_state.ydim = int(len(object_view[i][obj_option][1][0]) / st.session_state.xdim)
            
            obj_2d_cols = st.columns([4, 2, 1])
            obj_option = None

            with obj_2d_cols[0]:
                obj_option = st.selectbox('Choose an object to view in 2D', obj_names)
                st.write('You selected:', obj_option)

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
                            reshaped_details.append(obj_data[0][0][ind_y*xdim:(ind_y*xdim)+xdim])

                        reshaped_data[-1].append(item)
                        ind_x += 1
                        if (ind_x == xdim):
                            ind_x = 0
                            ind_y += 1

                    fig = go.Figure(data=go.Heatmap(z=reshaped_data, coloraxis="coloraxis", name=key, 
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
            cols = st.columns([1 for i in range(gpu_num-1)])
            other_gpus = [i for i in range(gpu_num)]
            other_gpus.remove(i)
            for peer_gpu in other_gpus:
                pair = "GPU" + str(i) + "-GPU" + str(peer_gpu)
                if pair in data_by_device:
                    cols[other_gpus.index(peer_gpu)].markdown(f"##### Data owner: **<span style='color:{pal[peer_gpu]} \
                                                              '> GPU{peer_gpu}</span>**", unsafe_allow_html=True)
                    table_objs = []
                    table_instr = []
                    table_lines = []
                    for item in data_by_device[pair].items():
                        if 'total' == item[0]:
                            table_instr.insert(0, item)    
                            table_objs.insert(0, item) 
                            table_lines.insert(0, item) 
                        elif '.E' in item[0]:
                            table_instr.append(item)                            
                        elif item[0].startswith('line'):
                            continue
                        else:
                            table_objs.append([data_by_obj[item[0].strip()]['var_name'], item[1]])
                    
                    for line in data_by_device[pair]['lines']:
                        table_lines.append([line[5:], data_by_device[pair][line]])
                    
                    df = pd.DataFrame.from_dict(table_objs)
                    df.columns = ['object', 'count']
                    cols[other_gpus.index(peer_gpu)].table(df.sort_values(by=['count'], ascending=False))

                    df = pd.DataFrame.from_dict(table_instr)
                    df.columns = ['instruction', 'count']
                    cols[other_gpus.index(peer_gpu)].table(df.sort_values(by=['count'], ascending=False))

                    df = pd.DataFrame.from_dict(table_lines)
                    df.columns = ['code line', 'count']
                    cols[other_gpus.index(peer_gpu)].table(df.sort_values(by=['count'], ascending=False))
                        # .scrollTop = ''' + str(graph_height + i/100) + ''';
            st.components.v1.html(scroll_js(graph_height + margin + i*0.0001), height=0)
            break


if __name__ == "__main__":
    
    st.set_page_config(layout="wide")

    if (len(sys.argv) < 2):
        print("Provide an input file")
    data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, keys, ops, addrs, ptx_code, ptx_code_rev = read_data(sys.argv[1])
    main()

    # with st.sidebar:
    #     st.markdown("""## SASS Instructions""")
    #     st.markdown("""<small><p style='margin-left:-10px;margin-bottom:-50px;padding-bottom:-50px;'>
    #                  <span style="color:Tomato;">SASS line</span> | 
    #                  <span style="color:Tomato;">Code line</span>   |   SASS Instruction</p></small>""",
    #         unsafe_allow_html=True)
    #     st.markdown(
    #         f'''
    #         <style>
    #             .css-1544g2n.e1fqkh3o4 {{
    #                 padding-top: 0px;
    #                 margin-top: 0px;
    #                 margin-right: -50px;
    #             }}
    #             .css-ysnqb2.egzxvld4 {{
    #                 {0}
    #                 margin-top: {0}rem;
    #                 padding-top: {0}rem;
    #                 padding-right: {0}rem;
    #                 padding-left: -60px;
    #                 margin-left: -60px;
    #                 padding-bottom: {0}rem;
    #             }}
    #         </style>
    #         ''',
    #         unsafe_allow_html=True)
        
    #     content_head = """<head><link rel="stylesheet"
    #     href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
    #     <script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    #     <script>hljs.highlightAll();</script></head><body><pre>"""
    #     content_end = """</pre></body>"""

    #     total_lines = len(ptx_code)
    #     index_chars = len(str(len(ptx_code)))+1
    #     if total_lines > 0:
    #         index_chars_src = len(str(ptx_code[-1][1]))
    #         line_index = 0
    #         for line, linenum in ptx_code:
    #             line_index+=1
    #             linenum = abs(linenum)
    #             ls = len(line) - len(line.lstrip())
    #             line = line.replace('<', '&lt')
    #             line = line.replace('>', '&gt')
    #             # st.code(str(line_index)+ '.  ' + line)
    #             content_head+="""<a id='ptxline""" + str(line_index) + """'><code class='hljs language-c'
    #                             style='padding-left:-100px;margin-bottom:0;padding-bottom:0;overflow-x:hidden"""
    #             if line_index == 1:
    #                 content_head+=";margin-top:-50px'>"
    #             else:
    #                 content_head+=";margin-top:0;padding-top:0'>"
    #             content_head += str(line_index)+ '.' + ''.join([' '*(index_chars-len(str(line_index)))]) \
    #                             + str(linenum)+ '.' + ''.join([' '*(index_chars_src-len(str(linenum)))]) + ' | ' + line + "</code></a>"
    #         clicked_ptx = click_detector(content_head + content_end)
    #     # print(clicked_ptx)
    #     # if 'ptxline' in clicked_ptx:
    #     #     st.components.v1.html(scroll_js_to_line(abs(ptx_code[int(clicked_ptx[7:])][1])), height=0)

    st.markdown("""---""")

    f = None
    if os.path.exists(src_code_file):
        f = open(src_code_file, "r")

    if f is None:
        print("Source code file not found")
        exit(0)


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
            href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
        <script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
        <script>hljs.highlightAll();</script>
        </head><body><pre>"""
    content_end = """</pre></body>"""

    max_line_comm = 0
    total_comm = 0

    for line_index in data_by_line.keys():
        total_comm += data_by_line[line_index]['total']
        if (data_by_line[line_index]['total'] > max_line_comm):
            max_line_comm = data_by_line[line_index]['total']

    # https://colorkit.co/palette/413344-614c65-806485-936397-a662a8-664972-463c57-6e8da9-
    # 91bcdd-567d99-395e77-305662-264d4d-315c45-8a9a65-b6b975-b65d54-b60033-98062d-800022/
    pal_base = sns.blend_palette(pal_base[:-3], n_colors=32)
    pal = sns.color_palette([scale_lightness(color, 1.2) for color in pal_base]).as_hex()
    pal.reverse()

    lines = f.readlines()
    total_lines = len(lines)
    index_chars = len(str(total_lines))+1
    line_index = 0
    for line in lines:
        line_index+=1
        ls = len(line) - len(line.lstrip())
        line = line.replace('<', '&lt')
        line = line.replace('>', '&gt')
        # st.code(str(line_index)+ '.  ' + line)
        # content_head+="""<div style='display:flex;flex-direction:row;><code class='hljs language-c'
        #                 style='width:90%;margin-bottom:0;padding-bottom:0;overflow-x:hidden"""
        comm_percentage = 0.0
        if data_by_line.get(line_index, None) is not None:
            comm_percentage = float(data_by_line[line_index]['total'])/float(total_comm)*100
        str_percentage = f"{comm_percentage:2.2f}%"
        if (comm_percentage < 10.0):
            str_percentage = " " + str_percentage
        if (comm_percentage == 0):
            str_percentage = " " * 6    
        div_inject = "<span style='display:inline-block;' >"+ str_percentage +"</span>"
        content_head+="""<div id='d""" + str(line_index) + """'style='position:relative;width=100%'>""" +div_inject+ """<a
                          id='line""" + str(line_index) + """' style='display:inline-block;'><code class='hljs language-c'
                          style='position:relative;width:100%; background-color:rgb(0,0,0,0);margin-bottom:0;
                          padding-bottom:0;overflow-x:hidden;"""
        #div_inject = "<div class='percentage' style='background-color:" + pal[int(comm_percentage)%len(pal)] + ";opacity:0.4;width:" + str(comm_percentage) + "%'></div>"
        #content_head+="""<div id='line""" + str(line_index) + """'style='position:relative;width=100%'>""" +div_inject+ """<a 
        #                id='line""" + str(line_index) + """'><code class='hljs language-c' style='position:relative;width:100%;
        #                background-color:rgb(0,0,0,0);margin-bottom:0;padding-bottom:0;overflow-x:hidden;"""
        # if line_index in ptx_code_rev.keys():
        #     print("THE LINE!")
        #     print(line_index)
        #     content_head+=";margin-top:50;padding-top:50;background:#6da2d1"
        if line_index == 1:
            content_head+="'>"
        else:
            content_head+=";margin-top:0;padding-top:0'>"
        content_head+=str(line_index)+ '.' + ''.join([' '*(index_chars-len(str(line_index)))]) + line + "</a></code>"
        # content_head+="<progress style='position:absolute;top:0;left:0;z-index:-1;' value='" + str(comm_percentage) + "' max='100'> 32% </progress></div>"


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
        linenum = int(clicked_src[4:])
        if (linenum in data_by_line):
            with st.sidebar:
                linedata = data_by_line[linenum]
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
                                <code class='hljs language-c'>""" + lines[linenum-1] + """</code></body>""",
                    unsafe_allow_html=True,)
                st.markdown("## Total transfers: " + str(linedata['total']))

                # obj_str = "### Object(s) involved: \\"
                # for obj in linedata.get('objects', set()):
                #     obj_str += str(obj) + ": " + str(linedata[obj]) + "" 

                # st.markdown(obj_str)
                
                cols_rows_name = ['GPU%d' % i for i in range(gpu_num)]

                widths = []

                for i in range(gpu_num):
                    src_label = "GPU"+str(i)
                    widths.append([])
                    for j in range(gpu_num):
                        target_label = "GPU"+str(j)
                        pair_label = src_label+"-"+target_label
                        width = 0.0
                        if (pair_label in linedata):
                            width = linedata[pair_label]
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
                fig.update_yaxes(title_standoff = 10)

                fig.layout.width = 300
                fig.layout.height = 300
                chosen_point = plotly_events(fig)
                # print(chosen_point)
                if len(chosen_point) > 0:
                    chosen_point = chosen_point[0]['pointNumber']
                else:
                    chosen_point = None
    # print(clicked_src)
