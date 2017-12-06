import cv2
import numpy as np
import logging
import socket
import argparse
from collections import deque
import imutils

#Declaracion de configuracion de la bitacora y configurando el tipo de retornos que hara en la bitacora
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='ServerDataAnalytic.log',
                    filemode='w')


#Se declara la funcion de tracking_image que estara encargada del analisis de las imagenes







def tracking_image():
    
    ap = argparse.ArgumentParser()
   
    
    ap.add_argument("-b", "--buffer", type=int, default=32,
        help="max buffer size")
    ap.add_argument("-m","--match", type=int, default=30,
        help="MIN_MATCH_COUNT" )
    ap.add_argument("-v", "--video",
    	help="path to the (optional) video file")

    args = vars(ap.parse_args())
  
    # initialize the list of tracked points, the frame counter,
    # and the coordinate deltas
    pts = deque(maxlen=args["buffer"])  
    counter = 0
    (dX, dY) = (0, 0)
    direction = ""

    #Se establece la cantidad maxima de caracteristicas en comun
    MIN_MATCH_COUNT= maxlen=args["match"]
    detector=cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDITREE=0
    flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
    flann=cv2.FlannBasedMatcher(flannParam,{})

    #Se carga la imagen la cual sera reconocida con el metodo imred()
    trainImg=cv2.imread("intel.png")

    #La imagen cargada se convierte a escala de grises para un mejor reconocimiento con el metodo cvtColor()
    gray = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)

    #Con el metodo detectAndCompute extraemos las caracteristicas principales de la imagen
    #Este regresa una tupla que se almacenan en dos variables mandandole como parametro la imagen en escala de grises
    #trainKP guarda los puntos clave y coordenadas de la imagen
    #trainDesc guarda los descriptores
    trainKP,trainDesc=detector.detectAndCompute(gray,None)

    # if a video path was not supplied, grab the reference
    # to the webcam
    if not args.get("video", False):
        cam = cv2.VideoCapture(0)
    
    # otherwise, grab a reference to the video file
    else:
        cam = cv2.VideoCapture(args["video"])

    kernel = np.ones((5,5),np.uint8)

    #Se hace un ciclo infinito para que haga la captura de los frames y se detiene hasta que presionemos la letra 'q'
    while True:

        #Toma el primer frame de la camara con read() y regresa una tupla
        #ret es de tipo bool y notifica si se estan capturando los frames
        #QueryImgBGR guarda los frames que se estan guardando
        ret, QueryImgBGR=cam.read()

        #Si ret esta leyendo los frames entonces procedemos a ejecutar el match de imagenes
        if ret == True:

            #Convertimos el frame capturado en escala de grises
            QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)

            #Detect and compute extrae las caracteristicas principales como las coordenadas y puntos clave asi como
            #Los descriptotes y regresa una tupla
            #queryKP guarda las coordenadas y puntos clave
            #queryDesc guarda los descriptores
            queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)

            #Utilizamos un metodo de match mandandole los descriptores
            matches=flann.knnMatch(queryDesc,trainDesc,k=2)

            #Ahora nos aseguramos de tener coincidencias verdaderas
            goodMatch=[]
            for m,n in matches:
                if(m.distance<0.75*n.distance):
                    goodMatch.append(m)
            if(len(goodMatch)>MIN_MATCH_COUNT):
                tp=[]
                qp=[]
                for m in goodMatch:
                    tp.append(trainKP[m.trainIdx].pt)
                    qp.append(queryKP[m.queryIdx].pt)
                tp,qp=np.float32((tp,qp))
                H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)

                #Con shape sacamos la altura y el ancho
                h,w,_=trainImg.shape

                #Guardamos los datos de la altura y el ancho en la bitacora
                logging.info("Valor de Altura:")
                logging.info(h)
                logging.info("Valor del Ancho")
                logging.info(w)

                #Se empieza hacer el cuadrado que cubrira a la imagen una vez ya detectada
                trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
                queryBorder=cv2.perspectiveTransform(trainBorder,H)
                cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)

                #Se empieza a calcular el centro de la imagen y las coordenadas
                rangomax=np.array([50,255,50])
                rangomin=np.array([0,51,0])
                mascara=cv2.inRange(QueryImgBGR, rangomin, rangomax)
                mascara = cv2.erode(mascara, None, iterations=2)
                mascara = cv2.dilate(mascara, None, iterations=2)
            
                # find contours in the mask and initialize the current
                # (x, y) center of the frame
                cnts = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
                center = None

                #print str(cnts)
                opening=cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
                x,y,w,h = cv2.boundingRect(opening)

                #Se hace el calculo de las coordenadas (x, y)
                coorner_x = (x + w / 2)
                coorner_y = (y + h / 2)


                logging.info("Valor de coordenadas (x , y): ")
                logging.info(str(coorner_x) + " " + str(coorner_y))

                #Se dibuja el centro de la imagen una vez ya calculado
                cv2.circle(QueryImgBGR,(x+w/2, y+h/2),5,(0,0,255),-1)


                # only proceed if at least one contour was found
                if len(cnts) > 0:
                    #print "cantidad de contornos: " + str(cnts)
                    # find the largest contour in the mask, then use
                    # it to compute the minimum enclosing circle and
                    # centroid
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # only proceed if the radius meets a minimum size
                    if radius > 10:
                        # draw the circle and centroid on the frame,
                        # then update the list of tracked points
                        cv2.circle(QueryImgBGR, (int(x), int(y)), int(radius),
                            (0, 255, 255), 2)
                        cv2.circle(QueryImgBGR, center, 5, (0, 0, 255), -1)
                        pts.appendleft(center)

                    print "x: " + str(x)    
                    print "y: " + str(y)    
                    print "radio: " + str(radius)
                    pts.appendleft(center)



                    # loop over the set of tracked points
                    for i in np.arange(1, len(pts)):
                        # if either of the tracked points are None, ignore
                        # them
                        if pts[i - 1] is None or pts[i] is None:
                            continue
                
                        # check to see if enough points have been accumulated in
                        # the buffer
                        if counter >= 32 and i == 1 and pts[-10] is not None:
                            # compute the difference between the x and y
                            # coordinates and re-initialize the direction
                            # text variables
                            dX = pts[-10][0] - pts[i][0]
                            dY = pts[-10][1] - pts[i][1]
                            (dirX, dirY) = ("", "")
                
                            # ensure there is significant movement in the
                            # x-direction
                            if np.abs(dX) > 20:
                                dirX = "East" if np.sign(dX) == 1 else "West"
                
                            # ensure there is significant movement in the
                            # y-direction
                            if np.abs(dY) > 20:
                                dirY = "North" if np.sign(dY) == 1 else "South"
                
                            # handle when both directions are non-empty
                            if dirX != "" and dirY != "":
                                direction = "{}-{}".format(dirY, dirX)
                                print "coords: " + str(direction)
                                logger.info(str(direction))
                            # otherwise, only one direction is non-empty
                            else:
                                direction = dirX if dirX != "" else dirY  

                            # otherwise, compute the thickness of the line and
                            # draw the connecting lines
                            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
                    # show the movement deltas and the direction of movement on
                    # the frame
                    print str(direction)
                    cv2.putText(QueryImgBGR, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255), 3)
                    cv2.putText(QueryImgBGR, "dx: {}, dy: {}".format(dX, dY),
                        (10, QueryImgBGR.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)


               
            #Si no existen suficientes coincidencias mandamos el numero de coincidencias
            else:
                print "No existen coincidencias de imagen- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)

            #Muestra en pantalla el frame que se captura con la variable QueryImgBGR
            cv2.imshow('Analizador de vision artificial',QueryImgBGR)

            #Si queremos salir del programa presionamos la letra q
            if cv2.waitKey(10)==ord('q'):
                break
        #Si no se pudo leer el frame mandara un posible error de camara por no poder leer el frame
        else:
            logging.error("No se pudo leer ningun frame, es posible que no este conectada la camara")
    cam.release()
    #Cerramos la ventana que esta haciendo la captura de los fames
    cv2.destroyAllWindows()

#Al ejecutar el programa se invoca el metodo tracking_image encargado de hacer la captura y
#analisis de los frame
if __name__ == "__main__":
    tracking_image()