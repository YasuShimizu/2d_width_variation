import main
import os

runs=["Case1"]
for cn in runs:

    os.system("del /Q "+cn+"\\*.png")
    os.system("del /Q "+cn+"\\*.mp4")    
    os.system("del /Q "+cn+"\\bed\\*.png")    
    os.system("del /Q "+cn+"\\png\\*.png")
    os.system("del /Q "+cn+"\\png_q\\*.png")

    main.nays2d(cn)

    os.system("copy /Y *.png "+cn)
    os.system("copy /Y *.mp4 "+cn)
    cmd="copy /Y png\\*.png "+cn+"\\png"
    os.system(cmd)
    cmd="copy /Y png_q\\*.png "+cn+"\\png_q"
    os.system(cmd)
    cmd="copy /Y bed\\*.png "+cn+"\\bed"
    os.system(cmd)