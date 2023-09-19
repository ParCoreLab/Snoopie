from includes.parser import *
import json

with open("./visualizer/testdata/2gpu_log.txt","r") as f:
    ops = set()
    read_data(f,"2gpu_log.txt",(5, ops))
    print("abc")

topl = OpInfoRow.table(), ObjIdRow.table(), ObjNameRow.table(), FunctionInfoRow.table(), CodeLineInfoRow.table() 

print(topl)