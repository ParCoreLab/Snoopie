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
from includes import argumentparser, filepicker_page
from includes.streamlit_globals import *



data_by_address = {}
data_by_device = {}
data_by_obj = {}
data_by_line = {}
src_lines = []
chosen_line = None
# addr_obj_map = {}


ops = set()
ops_to_display = []

GRAPH_SHAPES = {"square": 0, "polygon": 1, "rectangle": 2}
MIN_TABLE_HEIGHT = 50
ROW_HEIGHT = 27
MAX_TABLE_HEIGHT = 540

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


def rectangle_coord(height, width, n):
    vertices_per_side = n/4
    result = []
    side_index = 0
    x_dim = 0
    y_dim = 0
    while (side_index < 4):
        side_count = 0
        delta_x = 0
        delta_y = 0
        if (side_index == 0):
            delta_x = width/vertices_per_side
        if (side_index == 1):
            delta_y = height/vertices_per_side
        if (side_index == 2):
            delta_x = -width/vertices_per_side
        if (side_index == 3):
            delta_y = -height/vertices_per_side
        while (side_count < vertices_per_side):
            side_count+=1
            x_dim += delta_x
            y_dim += delta_y
            result.append([x_dim, y_dim])
        side_index+=1
    return result


def isInt_try(v):
    try:     i = int(v)
    except:  return False
    return True


def detect_gpu_count(f):
    global gpu_num

    skip = True
    count = set()
    for line in f:
        if skip:
            skip = False; continue
        if line.startswith("offset") or line.startswith("code_line_index"):
            break
        keys = line.split(",")
        running_dev = int(keys[3])
        mem_dev = int(keys[4])
        if running_dev >= 0:
            count.add(running_dev)
        if mem_dev >= 0:
            count.add(mem_dev)
    gpu_num = len(count)
    print("gpu num detected as:", gpu_num)
    f.seek(0)
    

@st.cache_data
def read_data(_file, filename):
    if _file == None or filename == None:
        st.experimental_rerun() # this shouldn't be here need to fix the problem soon
    global data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, ops
    file = _file
    if gpu_num <= 0:
        detect_gpu_count(file)

    addrs = set()
    
    # prints all files
    graph_name = ""
    pickle_file = None
            
    pickle_filename = ''.join(filename.split(".")[:-1]) + '.pkl'
    if os.path.isfile(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            pickle_file = pickle.load(f)
            if len(pickle_file) > 6:
                new_pickle_file = pickle_file[:5]
                new_pickle_file.append(pickle_file[6])
                pickle_file = new_pickle_file
            for item in pickle_file:
                pass
                # print(item)
                # print()
            print("Data loaded from " + pickle_filename)

    reading_data = 0
    opkeys = []
    objkeys = []
    counter = 0

    if (pickle_file is None):    
        
        for line in file:
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
                mem_range = data.get("mem_range", 4)

                if "U8" in operation:
                    mem_range = 4
                if "U16" in operation:
                    mem_range = 8
                elif "32" in operation:
                    mem_range = 4
                elif "64" in operation:
                    mem_range = 8
                elif "128" in operation:
                    mem_range = 16

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

                temp_data[device] = temp_data.get(device, {})
                temp_data[device][operation] = temp_data[device].get(operation, 0) + 1
                
                temp_data['line_' + str(linenum)] = temp_data.get('line_' + str(linenum), {})
                temp_data['line_' + str(linenum)][operation] = temp_data['line_' + str(linenum)].get(operation, 0) + 1

                temp_lines = temp_data.get('lines', set())
                temp_lines.add('line_' + str(linenum))
                temp_data['lines'] = temp_lines

                data_by_device[pair] = data_by_device.get(pair, {})
                temp_data = data_by_device[pair]
                temp_data['total'] = temp_data.get('total', 0) + 1
                temp_data['totalbytes'] = temp_data.get('totalbytes', 0) + mem_range
                temp_data[operation] = temp_data.get(operation, 0) + 1
                temp_data[operation + "bytes"] = temp_data.get(operation + "bytes", 0) + mem_range
                temp_data[obj_name] = temp_data.get(obj_name, 0) + 1
                temp_data['line_' + str(linenum)] = temp_data.get('line_' + str(linenum), 0) + 1
                temp_lines = temp_data.get('lines', set())
                temp_lines.add('line_' + str(linenum))
                temp_data['lines'] = temp_lines
                
                data_by_device[device] = data_by_device.get(device, {})
                temp_data = data_by_device[device]
                temp_data['totalbytes'] = temp_data.get('totalbytes', 0) + mem_range
                temp_data['total'] = temp_data.get('total', 0) + 1
                temp_data[operation] = temp_data.get(operation, 0) + 1
                temp_data[operation + "bytes"] = temp_data.get(operation + "bytes", 0) + mem_range
                temp_data[owner] = temp_data.get(owner, 0) + 1
                temp_data[owner+'bytes'] = temp_data.get(owner+'bytes', 0) + mem_range
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

        for line in file:
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

        print("Reading complete")

        all_data = [data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, ops]
        
        with open(pickle_filename, 'wb') as pf:
            pickle.dump(all_data, pf)
            print("Data saved to " + pickle_filename)
    else:
        data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, ops = pickle_file

    return data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, ops


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = min(0.85, s * 1.75) )


def get_object_view_data(filter = None, allowed_ops = []):
    
    def check_filter(hex_addr) -> bool:
        nonlocal filter, allowed_ops
        if hex_addr not in data_by_address: return False
        if filter == None:
            # allow all gpus
            for op in allowed_ops:
                if op in data_by_address[hex_addr]: return True
            return False
        if filter not in data_by_address[hex_addr]: return False
        for op in allowed_ops:
            if op in data_by_address[hex_addr][filter]: return True
        return False


    cols_rows_name = ['GPU%d' % i for i in range(gpu_num)]
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
            if check_filter(hex_addr):
                dict_details = data_by_address[hex_addr]
                html_details = "<br>"
                for dev in cols_rows_name:
                    if dev in dict_details:
                        html_details += " "*8 + str(dev) + ": " + str(dict_details[dev]) + "<br>"
                for op in ops:
                    if op in dict_details:
                        html_details += " "*8 + str(op) + ": " + str(dict_details[op]) + "<br>"
                for line in dict_details['lines']:
                    intline = int(line[5:])
                    html_details += " "*8 + str(line) + ": " + str(dict_details[line]) + "<br>"
                    data_by_line[intline] = data_by_line.get(intline, {})
                    temp_data = data_by_line[intline]
                    # calculate how many times each object is updated in that line
                    temp_line_total = 0
                    for op in allowed_ops:
                        temp_line_total += dict_details.get(line, {}).get(op, 0)

                    temp_data["objects_updated"] = temp_data.get("objects_updated", dict())
                    temp_data["objects_updated"][data_by_obj[key]["var_name"]] = temp_data["objects_updated"].get(data_by_obj[key]["var_name"], 0) + temp_line_total
                obj[-1].append([hex_addr, html_details])
                temp_total = 0
                if filter == None:
                    for op in allowed_ops:
                        temp_total += dict_details.get(op,0)
                else:
                    for op in allowed_ops:
                        temp_total += dict_details.get(filter,{}).get(op,0)
                object_map[-1].append(temp_total)
            else:
                obj[-1].append([hex_addr, ""])
                object_map[-1].append(0)
        if (data_by_obj[key]['device_id'] >= 0):
            object_view[data_by_obj[key]['device_id']][data_by_obj[key]['var_name']] = [obj, object_map]
    return object_view

def main():
    global data_by_address, data_by_device, gpu_num, ops, chosen_line, ops_to_display

    top_cols = st.columns([5,5])

    with top_cols[0]:
        st.radio(
            "Communication units",
            ["Data transfers", "Bytes"],
            key="units",
            horizontal=True
        )
    with top_cols[1]:
        ops_to_display = st.multiselect("Operations to display", options = ops, default = ops)


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

    #graph_width = 1200
    graph_width = 600
    #graph_height = 800
    graph_height = 400
    font_size = int(graph_height/22)
    margin = 30

    graph_shape = GRAPH_SHAPES["polygon"]
    positions = []

    if (graph_shape == GRAPH_SHAPES["polygon"]):
        positions = regular_polygon_coord([int(graph_width/2), int(graph_height/2)], 
                        int(min(graph_width, graph_height)/2)-margin, gpu_num)
    elif (graph_shape == GRAPH_SHAPES["square"]):
        positions = rectangle_coord(graph_height, graph_height, gpu_num) 
    elif (graph_shape == GRAPH_SHAPES["rectangle"]):
        positions = rectangle_coord(graph_height, graph_width, gpu_num) 

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
        if (graph_shape == GRAPH_SHAPES["polygon"]):
            for pos in positions:
                pos[0] += ratio*(pos[0]-graph_width/2)/2
                pos[0] = max(margin, pos[0])
                pos[0] = min(graph_width-margin, pos[0])
        

    cols = math.ceil(math.sqrt(gpu_num))
    rows = int(gpu_num/cols)
    # delta_pos = [int((graph_width-2*margin)/max(1, cols-1)), int((graph_height-2*margin)/max(1, rows-1))]

    label_bytes = ""

    if (st.session_state.units == "Bytes"):
        label_bytes = "bytes"


    for i in range(gpu_num):
        label = "GPU"+str(i)

        size = 0.0
        if (label in data_by_device):
            for optype in ops_to_display:
                size += data_by_device[label][optype + label_bytes]
        sizes.append(size)
    
    norm_ratio = max(sizes)/max_size
    if norm_ratio == 0: norm_ratio = 1

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
            pair = src_label + "-" + target_label
            width = 0.0
            if (pair in data_by_device):
                for op in ops_to_display:
                    if op in data_by_device[pair]:
                        width += data_by_device[pair][op + label_bytes]*sampling_period
            widths[i].append(width)
            if width > max_val:
                max_val = width
    norm_ratio = max(max(widths))/max_width
    if norm_ratio == 0: norm_ratio = 1
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
    #cols = st.columns([100, 1])
            
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
        fig.update_yaxes(title_standoff = 10)

        fig.layout.width = graph_width*(4.0/7.0)
        fig.layout.height = graph_height
        chosen_point = plotly_events(fig)
        if len(chosen_point) > 0:
            chosen_point = chosen_point[0]['pointNumber']
        else:
            chosen_point = None

    if chosen_point != None:
        chosen_id_graph = chosen_point[0]

    chosen_id_tab = stx.tab_bar(data=[
        stx.TabBarItemData(id=str(i), title="GPU"+str(i), description="") for i in range(gpu_num)], default=0)

    object_view = []
    for dev in range(gpu_num):
        object_view.append({})

    selected_rows = None


    object_view = get_object_view_data(filter = None, allowed_ops = ops_to_display)
    

    for i in range(gpu_num):
        if len(object_view[i]) > 0 and chosen_id_tab == str(i):
            st.markdown(f"### Objects owned by **<span style='color:{pal[i]}'>GPU{i}</span>**", unsafe_allow_html=True)
            
            objects_owned_cols = st.columns([4,7])
            other_gpus_selector = ["All Accesses"] + [s for s in cols_rows_name if s != f"GPU{i}"]
            with objects_owned_cols[0]:
                filter_chooser = st.selectbox("Filter accesses by GPU",other_gpus_selector, index = 0)

            is_all_zeros = False
            if filter_chooser != other_gpus_selector[0]:
                object_view = get_object_view_data(filter_chooser, allowed_ops = ops_to_display)
                is_all_zeros = True
                for key in object_view[i].keys():
                    obj_data = object_view[i][key]
                    for arr in obj_data[1]:
                        for elem in arr:
                            if elem != 0:
                                is_all_zeros = False
                            if not is_all_zeros: break
                        if not is_all_zeros: break
                    if not is_all_zeros: break
        

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

            if is_all_zeros:
                obj_fig.update_layout(
                    {
                        "coloraxis_cmin": -.2,
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
                    width=graph_width*(3/2)
                )
            obj_fig['layout']['xaxis' + str(index-1)].update(dict(title="Offset", title_standoff=8))
                
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
            cols = st.columns([1 for i in range(3)])
            other_gpus = [i for i in range(gpu_num)]
            other_gpus.remove(i)
            table_objs = []
            table_instr = []
            table_lines = []
            for peer_gpu in other_gpus:
                pair = "GPU" + str(i) + "-GPU" + str(peer_gpu)
                if pair in data_by_device:
                    for item in data_by_device[pair].items():
                        litem = [item[0], item[1], peer_gpu]
                        if 'total' == item[0]:
                            table_instr.insert(0, litem)    
                            table_objs.insert(0, litem) 
                            # table_lines.insert(0, litem) 
                        elif '.E' in item[0]:
                            if not item[0].endswith("bytes"):
                                table_instr.append(litem)                            
                        elif item[0].startswith('line') or item[0].startswith('totalbytes'):
                            continue
                        else:
                            table_objs.append([data_by_obj[item[0].strip()]['var_name'], item[1], peer_gpu])
                    
                    for line in data_by_device[pair]['lines']:
                        table_lines.append([int(line[5:]), data_by_device[pair][line], peer_gpu])

            ag_custom_css={
                    "#gridToolBar": {
                        "padding-bottom": "0px !important",
                    }
                }
                    
            
            cols[0].markdown(f"##### Objects", unsafe_allow_html=True)
            df = pd.DataFrame.from_dict(table_objs)
            df.columns = ['object', 'count', 'destination']
            table_height = min(MIN_TABLE_HEIGHT + len(df) * (ROW_HEIGHT), MAX_TABLE_HEIGHT)
            with cols[0]:
                AgGrid(df, height=table_height, fit_columns_on_grid_load=True, custom_css=ag_custom_css,
                                        columns_auto_size_mode = ColumnsAutoSizeMode.FIT_CONTENTS)

            cols[1].markdown(f"##### Instructions", unsafe_allow_html=True)
            df = pd.DataFrame.from_dict(table_instr)
            df.columns = ['instruction', 'count', 'destination']
            table_height = min(MIN_TABLE_HEIGHT + len(df) * (ROW_HEIGHT), MAX_TABLE_HEIGHT)
            with cols[1]:
                AgGrid(df, height=table_height, fit_columns_on_grid_load=True, custom_css=ag_custom_css,
                                        columns_auto_size_mode = ColumnsAutoSizeMode.FIT_CONTENTS)

            cols[2].markdown(f"##### Code lines", unsafe_allow_html=True)
            df = pd.DataFrame.from_dict(table_lines)
            df.columns = ['code line', 'count', 'destination']
            table_height = min(MIN_TABLE_HEIGHT + len(df) * (ROW_HEIGHT), MAX_TABLE_HEIGHT)
            with cols[2]:
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_selection('single', use_checkbox=False, pre_selected_rows=None)
                gridOptions = gb.build()
                grid_response = AgGrid(df, gridOptions, height=table_height, fit_columns_on_grid_load=True,
                                        update_mode = GridUpdateMode.SELECTION_CHANGED, custom_css=ag_custom_css,
                                        columns_auto_size_mode = ColumnsAutoSizeMode.FIT_CONTENTS)
                selected_rows = grid_response['selected_rows']
                if (len(selected_rows) > 0):
                    if (selected_rows[0]['code line'] != 'total'):
                        chosen_line = int(selected_rows[0]['code line'])
                        tkey = 'table_select' + str(peer_gpu)
                        if (tkey not in st.session_state or st.session_state[tkey] != chosen_line):
                            show_sidebar()
                            st.session_state[tkey] = chosen_line
                        # show_sidebar(int(selected_rows['code line']))
                        # .scrollTop = ''' + str(graph_height + i/100) + ''';
            components.html(scroll_js(graph_height + margin + i*0.0001), height=0)
            break


def show_sidebar():
    global chosen_line
    # print("CHOSEN LINE " + str(chosen_line))
    linenum = chosen_line
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
                            <code class='hljs language-c'>""" + src_lines[linenum-1] + """</code></body>""",
                unsafe_allow_html=True,)
            st.markdown("### Total transfers: " + str(linedata['total']))
            st.markdown("#### Objects updated: ")
            print(linedata)
            for key in linedata['objects_updated'].keys():
                st.markdown("###### " + key + ": "  + str(linedata['objects_updated'][key]))

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
            st.session_state.sidebar_state = 'expanded'

def read_code(f):
    global src_lines
    f.seek(0)
    src_lines = f.readlines()


def show_code():
    global src_lines, chosen_line, ops_to_display
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
    filtered_comm = {}

    for line_index in data_by_line.keys():
        ftotal = 0
        for op in ops_to_display:
            ftotal += data_by_line[line_index].get(op, 0)
        filtered_comm[line_index] = ftotal
        total_comm += ftotal
        if (ftotal > max_line_comm):
            max_line_comm = ftotal

    if total_comm == 0: total_comm = 1 # division by zero
    # https://colorkit.co/palette/413344-614c65-806485-936397-a662a8-664972-463c57-6e8da9-
    # 91bcdd-567d99-395e77-305662-264d4d-315c45-8a9a65-b6b975-b65d54-b60033-98062d-800022/
    pal_base_lines = sns.blend_palette(pal_base[:-3], n_colors=32)
    pal = sns.color_palette([scale_lightness(color, 1.2) for color in pal_base_lines]).as_hex()
    pal.reverse()

    total_lines = len(src_lines)
    index_chars = len(str(total_lines))+1
    line_index = 0
    for line in src_lines:
        line_index+=1
        ls = len(line) - len(line.lstrip())
        line = line.replace('<', '&lt')
        line = line.replace('>', '&gt')
        # st.code(str(line_index)+ '.  ' + line)
        # content_head+="""<div style='display:flex;flex-direction:row;><code class='hljs language-c'
        #                 style='width:90%;margin-bottom:0;padding-bottom:0;overflow-x:hidden"""
        comm_percentage = 0.0
        if data_by_line.get(line_index, None) is not None:
            comm_percentage = float(filtered_comm[line_index])/float(total_comm)*100
        div_inject = "<div class='percentage' style='background-color:" + pal[int(comm_percentage)%len(pal)] + ";opacity:0.4;width:" + str(comm_percentage) + "%'></div>"
        
        str_percentage = f"{comm_percentage:2.2f}%"
        if (comm_percentage < 10.0):
            str_percentage = " " + str_percentage
        if (comm_percentage == 0):
            str_percentage = " " * 6
        div_inject += "<span style='position:absolute; bottom:0.2rem;'>"+ str_percentage +"</span>"
        content_head+="""<div id='d""" + str(line_index) + """'style='position:relative;width=100%'>""" +div_inject+ """<a 
                        id='line""" + str(line_index) + """' style='display:inline-block;margin-left:2.5rem'><code class='hljs language-c' style='position:relative;width:100%;
                        background-color:rgb(0,0,0,0);margin-bottom:0;padding-bottom:0;overflow-x:hidden;"""
        if line_index == 1:
            content_head+="'>"
        else:
            content_head+=";margin-top:0;padding-top:0'>"
        content_head+=str(line_index)+ '.' + ''.join([' '*(index_chars-len(str(line_index)))]) + line + "</a></code></div>"


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
        ckey = 'code_select'
        if (ckey not in st.session_state or st.session_state[ckey] != chosen_line):
            show_sidebar()
            st.session_state[ckey] = chosen_line



def continue_main():
    global data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, ops

    data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, ops = read_data(logfile, logfile_name)

    read_code(src_code_file)
    main()

    st.markdown("""---""")

    show_code()


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    argumentparser.parse()
    setup_globals()
    if not st.session_state.show_filepicker:
        start_newfile = st.button("Profile another file")
        if start_newfile:
            st.session_state.show_filepicker = True

    if st.session_state.show_filepicker:
        if filepicker_page.filepicker_page() == False:
            continue_main()
    else:
        continue_main()
    # print(clicked_src)
