from roboflow import Roboflow
rf = Roboflow(api_key="fsci1KhoH9BF4H1HF3Vj")
project = rf.workspace("034-ganesh-kumar-m-v-cs-r2lwe").project("basketball-lhqoe")
dataset = project.version(1).download("yolov8")