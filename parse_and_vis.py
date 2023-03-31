import sys
import math
import os
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import pickle 
import extra_streamlit_components as stx
# from st_click_detector import click_detector
import seaborn as sns
import pandas as pd
import colorsys
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go


data_by_address = {}
data_by_device = {}
data_by_obj = {}
addr_obj_map = {}
gpu_num = 0
keys = []
ops = []
addrs = []
ADDR_DIST = 4


def scroll_js(top):
    return '''<script>
                var body = window.parent.document.querySelector(".main");
                body.scrollTo({top: ''' + str(top) + ''', behavior: 'smooth'});
            </script>'''


def regular_polygon_coord(center, radius, n):
    return [[center[0] + radius * math.sin((2*math.pi/n) * i),
            center[1] + radius * math.cos((2*math.pi/n) * i)] for i in range(n)]


@st.cache_data
def read_data(file):
    global data_by_address, data_by_device, data_by_obj, gpu_num, keys, ops, addrs
    f = open(file, "r")

    if f is None:
        print("File not found")
        exit(0)
    
    # prints all files
    graph_name = ""
    all_data = []
    pickle_file = None
            
    pickle_filename = ''.join(file.split(".")[:-1]) + '.pkl'
    if os.path.isfile(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            pickle_file = pickle.load(f)
            print("Data loaded from " + pickle_filename)

    if (pickle_file is None):    
        for line in f:
            if (line.startswith("filename=")):
                splt_info = line.split(',')
                graph_name = splt_info[0].split('/')[-1]
                gpu_num = int(splt_info[1][-1])
            if (line.startswith('{"op": "mem_')):
                continue
            if (line.startswith('{"op"')):
                opdata = {}
                splt_info = line[1:-2].split(',')
                if len(keys) == 0:
                    for splt in splt_info:
                        keys.append(splt.split(':')[0].strip()[1:-1])
                for i in range(len(splt_info)):
                    opd = splt_info[i].split(':')[1].strip()
                    if '"' in opd:
                        opd = opd[1:-1]
                    else:
                        opd = int(opd)
                    opdata[keys[i]] = opd
                all_data.append(opdata)
            if (line.startswith('offset:')):
                obj_data = {}
                hex_off = ''
                int_off = 0
                for splt in line.split(','):
                    print(splt)
                    kv = splt.strip().split(' ')
                    key = kv[0][:-1]
                    if 'name' in key:
                        value = kv[1].split('[')[0]
                    elif key == 'offset':
                        int_off = int(kv[1])
                        hex_off = str.format('0x{:016x}', int(kv[1]))
                        value = int(kv[1]) 
                    else:
                        value = int(kv[1]) 
                    obj_data[key] = value
                data_by_obj[hex_off]=obj_data
                for i in range(obj_data['size']):
                    hex_addr = str.format('0x{:016x}', int_off+(i*4))
                    addr_obj_map[hex_addr] = hex_off

        print(data_by_obj)
        print("Reading complete")

        for data in all_data:
            address = data["addr"]
            operation = data["op"]
            obj_name = data_by_obj[addr_obj_map[address]]['var_name']
            if address not in addrs:
                addrs.append(address)
            if operation not in ops:
                ops.append(operation)
            device = "GPU"+str(data["running_device_id"])
            owner = "GPU"+str(data["mem_device_id"])
            pair = device+"-"+owner

            data_by_address[address] = data_by_address.get(address, {})
            temp_data = data_by_address[address]
            temp_data['total'] = temp_data.get('total', 0) + 1
            temp_data[operation] = temp_data.get(operation, 0) + 1
            temp_data[device] = temp_data.get(device, 0) + 1

            data_by_device[pair] = data_by_device.get(pair, {})
            temp_data = data_by_device[pair]
            temp_data['total'] = temp_data.get('total', 0) + 1
            temp_data[operation] = temp_data.get(operation, 0) + 1
            temp_data[obj_name] = temp_data.get(obj_name, 0) + 1
            
            data_by_device[device] = data_by_device.get(device, {})
            temp_data = data_by_device[device]
            temp_data['total'] = temp_data.get('total', 0) + 1
            temp_data[operation] = temp_data.get(operation, 0) + 1
            temp_data[owner] = temp_data.get(owner, 0) + 1
            temp_data[obj_name] = temp_data.get(obj_name, 0) + 1

            
        print(graph_name, end=',')
        print(gpu_num)
        # for key, value in sorted(data_by_device.items(), key=lambda x: x[0]): 
        #     print("{} : {}".format(key, value))
        print(keys)
        print(ops)
        all_data = [data_by_address, data_by_device, data_by_obj, gpu_num, keys, ops, addrs]
        
        with open(pickle_filename, 'wb') as pf:
            pickle.dump(all_data, pf)
            print("Data saved to " + pickle_filename)
    else:
        data_by_address, data_by_device, data_by_obj, gpu_num, keys, ops, addrs = pickle_file

    return data_by_address, data_by_device, data_by_obj, gpu_num, keys, ops, addrs


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
    margin = 20
    positions = regular_polygon_coord([int(graph_width/2), int(graph_height/2)], 
                                      int(min(graph_width, graph_height)/2)-margin, 4)

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
    rows = int(gpu_num/cols)
    delta_pos = [int((graph_width-2*margin)/(cols-1)), int((graph_height-2*margin)/(rows-1))]

    for i in range(gpu_num):
        label = "GPU"+str(i)

        size = 0.0
        sizes.append(data_by_device[label]['total'])
    
    norm_ratio = max(sizes)/max_size

    for i in range(gpu_num):
        label = "GPU"+str(i)
        nodes.append(Node(id=i, color=pal[i%len(pal)], title=str(sizes[i]), label=label, font={"vadjust": -int(sizes[i]/norm_ratio+15)}, 
                          size=int(sizes[i]/norm_ratio), x=positions[i][0], y=positions[i][1]))
        
    widths = []
    for i in range(gpu_num):
        src_label = "GPU"+str(i)
        widths.append([])
        for j in range(gpu_num):
            target_label = "GPU"+str(j)
            width = 0.0
            if (target_label in data_by_device[src_label]):
                width = data_by_device[src_label][target_label]
            widths[i].append(width)
    
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
            edges.append(Edge(source=i, target=j, hidden=widths[i][j]==0, smooth={"enabled": True, "type": type, "roundness": 0.15}, 
                              color=pal2[i%len(pal2)], type=type, label=str(widths[i][j]), width=int(widths[i][j]/norm_ratio))) 

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
    
    # modify_cell((H0: 0)) 2  != (H1: 0) 0
    chosen_point = None
    with cols[1]:
        df = pd.DataFrame(widths, columns=cols_rows_name, index=cols_rows_name).astype('int')
        fig = px.imshow(df, color_continuous_scale='viridis',
                labels=dict(x="Owner", y="Issued by", color="Instruction count"))
        fig.update_traces(colorbar=dict(lenmode='fraction', len=0.5, thickness=10))
        fig.update_layout(
            font_family="Open Sans, sans-serif",
            font_color="#fafafa",
            paper_bgcolor="#0e1117"
        )
        fig.update_yaxes(title_standoff = 10)

        fig.layout.height = graph_height
        chosen_point = plotly_events(fig)
        print(chosen_point)
        if len(chosen_point) > 0:
            chosen_point = chosen_point[0]['pointNumber']
        else:
            chosen_point = None
        print(chosen_point)

    print(chosen_id_graph)
    if chosen_point != None:
        chosen_id_graph = chosen_point[0]
    print(chosen_id_graph)

    chosen_id_tab = stx.tab_bar(data=[
        stx.TabBarItemData(id=str(i), title="GPU"+str(i), description="") for i in range(gpu_num)], default=chosen_id_graph)

    object_view = []
    for dev in range(gpu_num):
        object_view.append({})

    for key in data_by_obj.keys():
        offset = data_by_obj[key]['offset']
        obj = [[]]
        object_map = [[]]
        obj_size = int(data_by_obj[key]['size']/ADDR_DIST)
        cols = int(math.sqrt(obj_size))*4
        ycounter = cols
        for i in range(0, int(data_by_obj[key]['size']), ADDR_DIST):
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

                obj[-1].append([hex_addr, html_details])
                object_map[-1].append(dict_details['total'])
            else:
                obj[-1].append([hex_addr, ""])
                object_map[-1].append(0)
        if (data_by_obj[key]['device'] >= 0):
            object_view[data_by_obj[key]['device']][data_by_obj[key]['var_name']] = [obj, object_map]
   
    for i in range(gpu_num):
        if chosen_id_tab == str(i):
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

                obj_fig.update_layout(coloraxis_colorbar=dict(title="Instruction<br>count"))               
                index += 1

            obj_fig.update_layout(coloraxis=dict(colorscale='viridis'), showlegend=False)
            obj_fig.update_layout(
                    margin=dict(l=120, r=120, t=20, b=60),
                    font_family="Open Sans, sans-serif",
                    font_color="#fafafa",
                    paper_bgcolor="#0e1117",
                )
            obj_fig['layout']['xaxis' + str(index-1)].update(dict(title="Offset", title_standoff=8))
                
            chosen_addr = plotly_events(obj_fig)
            print(chosen_addr)

            st.markdown("""---""")

            if 'ydim' not in st.session_state:
                st.session_state['ydim'] = 1
            if 'xdim' not in st.session_state:
                st.session_state['xdim'] = 1

            def calc_dim_y(obj_size):
                st.session_state.ydim = int(obj_size / st.session_state.xdim)
            
            obj_2d_cols = st.columns([4, 2, 1])
            obj_option = None

            with obj_2d_cols[0]:
                obj_option = st.selectbox('Choose an object to view in 2D', obj_names)
                st.write('You selected:', obj_option)
            with obj_2d_cols[1]:
                chosen_obj_size = len(object_view[i][obj_option][1][0])
                xdim = st.number_input('X-dimension', value=chosen_obj_size, min_value=1, on_change=calc_dim_y(chosen_obj_size), key='xdim')
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
                    ind_x = 0
                    for item in obj_data[1][0]:
                        if (ind_x == 0):
                            reshaped_data.append([])

                        reshaped_data[-1].append(item)
                        print(xdim)
                        ind_x += 1
                        if (ind_x == xdim):
                            ind_x = 0
                    print(reshaped_data)

                    fig = go.Figure(data=go.Heatmap(z=reshaped_data, coloraxis="coloraxis", name=key,
                        # hovertemplate="Object=%s<br>Offset=%%{x}<br>Instructions=%%{z}<br> \
                        #  Custom=%{customdata[0]}<extra></extra>"% key), row=index, col=1)
                        hovertemplate="<br>".join([
                            "X-offset: %{x}",
                            "Y-offset: %{y}",
                            "Instructions: %{z}"
                    ])))
                    fig.update_layout(coloraxis_colorbar=dict(title="Instruction<br>count"))               
                    index += 1

                    fig.update_layout(coloraxis=dict(colorscale='viridis'), showlegend=False)
                    fig.update_layout(
                            margin=dict(l=120, r=120, t=20, b=60),
                            font_family="Open Sans, sans-serif",
                            font_color="#fafafa",
                            paper_bgcolor="#0e1117",
                            yaxis=dict(autorange="reversed")
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
                    cols[other_gpus.index(peer_gpu)].table(dict(sorted(data_by_device[pair].items(), reverse=True)))
                        # .scrollTop = ''' + str(graph_height + i/100) + ''';
            st.components.v1.html(scroll_js(graph_height + margin + i*0.0001), height=0)
            break


if __name__ == "__main__":
    
    st.set_page_config(layout="wide")

    if (len(sys.argv) < 2):
        print("Provide an input file")
    data_by_address, data_by_device, data_by_obj, gpu_num, keys, ops, addrs = read_data(sys.argv[1])
    main()

    st.markdown("""---""")
    # f = open("workq_ring.cu", "r")

    # if f is None:
    #     print("Source code file not found")
    #     exit(0)

    # content_head = """<body><pre>"""
    # content_end = """</pre></body>"""

    # line_index = 0
    # lines = f.readlines()
    # total_lines = len(lines)
    # index_chars = len(str(total_lines))+1
    # for line in lines:
    #     line_index+=1
    #     ls = len(line) - len(line.lstrip())
    #     line = line.replace('<', '&lt')
    #     line = line.replace('>', '&gt')
    #     # st.code(str(line_index)+ '.  ' + line)
    #     if line_index == 1:
    #         content_head+="""<a id='line""" + str(line_index) + """'><code class="hljs language-c" 
    #                             style="margin-bottom:0;padding-bottom:0;overflow-x:hidden">""" \
    #                             + str(line_index)+ '.' + ''.join([' '*(index_chars-len(str(line_index)))]) \
    #                             + line + """</code></a>"""
    #     else:
    #         content_head+="""<a id='line""" + str(line_index) + """'><code class="hljs language-c" style="margin-top:0;
    #                             margin-bottom:0;padding-top:0;padding-bottom:0;overflow-x:hidden" >""" \
    #                             + str(line_index)+ '.' + ''.join([' '*(index_chars-len(str(line_index)))]) \
    #                             + line + """</code></a>"""

    # clicked = click_detector(content_head + content_end)
    # print(clicked)