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
                    filename='Tracking2D.log',
                    filemode='w')


#Se declara la funcion de find_coords para encontrar las coordenadas del centro
def find_coords( coordenada ):
    logging.info("chistorra")
    coordenadas_definidas =  [(0,0), (0,-1), (0,-2), (0,-3), (0,-4), (0,-5), (0,-6), (0,-7), (0,-8), (0,-9), (0,-10),
                             (-1,0), (-1,1), (-1,2), (-1,3), (-1,4), (-1,5), (-1,6), (-1,7), (-1,8), (-1,9), (-1,10), 
                             (-2,0), (-2,1), (-2,2), (-2,3), (-2,4), (-2,5), (-2,6), (-2,7), (-2,8), (-2,9), (-2,10),
                             (-3,0), (-3,1), (-3,2), (-3,3), (-3,4), (-3,5), (-3,6), (-3,7), (-3,8), (-3,9), (-3,10),
                             (-4,0), (-4,1), (-4,2), (-4,3), (-4,4), (-4,5), (-4,6), (-4,7), (-4,8), (-4,9), (-4,10),
                             (-5,0), (-5,1), (-5,2), (-5,3), (-5,4), (-5,5), (-5,6), (-5,7), (-5,8), (-5,9), (-5,10),
                             (-6,0), (-6,1), (-6,2), (-6,3), (-6,4), (-6,5), (-6,6), (-6,7), (-6,8), (-6,9), (-6,10),
                             (-7,0), (-7,1), (-7,2), (-7,3), (-7,4), (-7,5), (-7,6), (-7,7), (-7,8), (-7,9), (-7,10),
                             (1,-1), (1,-2), (1,-3), (1,-4), (1,-5), (1,-6), (1,-7), (1,-8), (1,-9), (1,-10),
                             (2,-1), (2,-2), (2,-3), (2,-4), (2,-5), (2,-6), (2,-7), (2,-8), (2,-9), (2,-10),
                             (3,-1), (3,-2), (3,-3), (3,-4), (3,-5), (3,-6), (3,-7), (3,-8), (3,-9), (3,-10),
                             (4,-1), (4,-2), (4,-3), (4,-4), (4,-5), (4,-6), (4,-7), (4,-8), (4,-9), (4,-10),                            
                             (5,-1), (5,-2), (5,-3), (5,-4), (5,-5), (5,-6), (5,-7), (5,-8), (5,-9), (5,-10),                            
                             (6,-1), (6,-2), (6,-3), (6,-4), (6,-5), (6,-6), (6,-7), (6,-8), (6,-9), (6,-10),                             
                             (7,-1), (7,-2), (7,-3), (7,-4), (7,-5), (7,-6), (7,-7), (7,-8), (7,-9), (7,-10),                             
                             ]
    coordenadas_obtenidas = coordenada
    for buffer_coordenadas in coordenadas_definidas:
        if  buffer_coordenadas in  coordenadas_obtenidas:
            print "Encontre el centro, RPAS listo para aterrizar"

#Se declara la funcion de tracking_image que estara encargada del analisis de las imagenes
def tracking_image():
    
    logging.info("****************************************")
    logging.info("******** Inicio de la bitacora *********")
    logging.info("****************************************")    
    ap = argparse.ArgumentParser()
   
    
    ap.add_argument("-b", "--buffer", type=int, default=32,
        help="max buffer size")
    ap.add_argument("-m","--match", type=int, default=30,
        help="MIN_MATCH_COUNT" )
    ap.add_argument("-v", "--video",
    	help="path to the (optional) video file")

    args = vars(ap.parse_args())
  
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

    if not args.get("video", False):
        cam = cv2.VideoCapture(0)
    
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

                # #Guardamos los datos de la altura y el ancho en la bitacora
                # logging.info("Valor de Altura:")
                # logging.info(h)
                # logging.info("Valor del Ancho")
                # logging.info(w)

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
            
                cnts = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]

                
                #print str(cnts)
                opening=cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
                x,y,w,h = cv2.boundingRect(opening)

                #Se hace el calculo de las coordenadas (x, y)
                # coorner_x = (x + w / 2)
                # coorner_y = (y + h / 2)


                # logging.info("Valor de coordenadas (x , y): ")
                # logging.info(str(coorner_x) + " " + str(coorner_y))

                #Se dibuja el centro de la imagen una vez ya calculado
                cv2.circle(QueryImgBGR,(x+w/2, y+h/2),5,(0,0,255),-1)


                if len(cnts) > 0:
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    if radius > 10:
                        logging.info("Valor de radio: ")
                        logging.info(radius)
                        cv2.circle(QueryImgBGR, (int(x), int(y)), int(radius),
                            (0, 255, 255), 2)
                        cv2.circle(QueryImgBGR, center, 5, (0, 0, 255), -1)
                        pts.appendleft(center)
                    try:
                        for i in np.arange(1, len(pts)):
                            logging.info("Longitud de puntos: ")
                            logging.info(str(len(pts)))
                            if pts[i - 1] is None or pts[i] is None:
                                continue
                            
                            if counter >= 10 and i == 1 and pts[-10] is not None:
                            
                                dX = pts[-10][0] - pts[i][0]
                                dY = pts[-10][1] - pts[i][1]
                                (dirX, dirY) = ("", "")
                    
                                if np.abs(dX) > 20:
                                    dirX = "is moving it Right" if np.sign(dX) == 1 else "is moving it Left"

                                if np.abs(dY) > 20:
                                    dirY = "is moving it Up" if np.sign(dY) == 1 else "is moving it Down"
                    
                                if dirX != "" and dirY != "":
                                    direction = "{}-{}".format(dirY, dirX)
                                    print "coords: " + str(direction)
                                    logging.info(str(direction))
                                else:
                                    direction = dirX if dirX != "" else dirY  

                                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                                cv2.line(QueryImgBGR, pts[i - 1], pts[i], (0, 0, 255), thickness)
                                
                                logging.info("Coordenada en X:")
                                logging.info(str(dX))
                                logging.info("Coordenada en Y: ")
                                logging.info(str(dY))
                                #Llama la funcion find_coords para mandar la coordenada obtenida y saber si esta en el centro
                                transformation_coords = str(dX) + "," + str(dY)
                                transformation_array_coords = eval('[(' + transformation_coords + ')]')
                                find_coords( transformation_array_coords )
                        cv2.putText(QueryImgBGR, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 255), 3)
                        cv2.putText(QueryImgBGR, "dx: {}, dy: {}".format(dX, dY),
                            (10, QueryImgBGR.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 0, 255), 1)
                        
                    except:
                        print "Fuera de rango"   
            #Si no existen suficientes coincidencias mandamos el numero de coincidencias
            else:
                print "No puedo encontrar el objeto"

            #Muestra en pantalla el frame que se captura con la variable QueryImgBGR
            cv2.imshow('Analizador de aterrizaje',QueryImgBGR)
            
            counter += 1
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