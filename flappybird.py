#drawing screen & input keyboard and mouse
import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui as pagui

#learning & making decision
import random
import numpy
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

#pagui.press('space')
#pagui.click()

#parameter for learning
LR = 1e-3
#goal_steps = 20
score_req = 13 #13
initial_games = 10000 #8000

#draw lines, test purpose.
def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img,(coords[0],coords[1]),(coords[2],coords[3]),[255,255,255],3)
    except:
        pass
#cut of the irrelvant region
def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    masked = cv2.bitwise_and(img, mask)
    return masked
        

def process_img(original_image, action, flag, num_of_space, time, time_increment, prev_obs, stop):

    #image processing for identifying lines
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    
    vertices = np.array([[320, 630],[320, 30],[550, 30],[550, 630]])
    processed_img = roi(processed_img,[vertices])

    kernel = np.ones((30, 30), np.uint8)
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 20, 15)
    #draw_lines(processed_img,lines)

    xmin1 = 0
    y11 = 0
    y12 = 0
    xmin2 =0
    y21 = 0
    y22 = 0
    num_of_lines = 0 


    if(action == 1):
        num_of_space += 1
    time += time_increment
    if time != 0:
        position =  num_of_space/time
    else:
        position = 0

    #parameter feedback for taking an action.
    state = (0,0,0,0,0,0,position) #(xmin1, y11, y12, xmin2, y21, y22,self position)
    reward = 0

    #try to find the 2 mostleft vertical lines and get their xy coordinates
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                num_of_lines += 1
                if(abs(x1-x2)>5):
                    continue
                elif(xmin1 == 0):
                    xmin1 = x1
                    y11 = y1
                    y12 = y2
                elif(xmin2 == 0):
                    xmin2 = x1
                    y21 = y1
                    y22 = y2
                elif(x1 < xmin1):
                    xmin1 = x1
                    y11 = y1
                    y12 = y2
                elif(x1 < xmin2):
                    xmin2 = x1
                    y21 = y1
                    y22 = y2
                else:
                    continue
        cv2.line(original_image,(xmin1,y11),(xmin1,y12),(0,0,255),10)
        cv2.line(original_image,(xmin2,y21),(xmin2,y22),(0,0,255),10)

        if(stop == 1):
            stop = 1
            done = True

        elif(flag == 1 and prev_obs[0] == xmin1 and prev_obs[1] == y11 and prev_obs[3] == xmin2 and prev_obs[4] == y21):
            done = True
            flag = 0
            #time = 0
            stop = 1
            #num_of_space = 0
        else:
            flag = 1
            state = (xmin1, y11, y12, xmin2, y21, y22, position)
            done = False
            #print('number of lines:', num_of_lines)
            #print('line position:', xmin1,' ', y11,' ', y12,' ', xmin2,' ', y21,' ', y22)
    except:
        #print('no vertical lines detected')
        if(flag == 1):
            flag = 0
            done = True
            #time = 0
            stop = 0
            #num_of_space = 0
        else:    
            done = False
            stop = 0
        pass
    if(flag==1):
        reward = 1

    return original_image, np.array(state), reward, done, flag , num_of_space, time, stop


#creating training data
def initial_population():

    #    
    training_data = []
    scores = []
    accepeted_scores = []

    #
    last_time = time.time()
    action = 1
    flag = 0
    time_increment = 0.0
    total_time = 0
    num_of_space = 0
    scores = 0
    prev_obs = np.array((0,0,0,0,0,0,0))
    stop = 0

    #
    games = 0
    helper = 0
    
    #
    game_memory = []
    
    while (games < initial_games):

        if flag == 1:
            action = random.randrange(0,2)
            if action == 1:
                pagui.press('space')
        elif helper == 0:
            helper += 1
            action = 1
            pagui.press('space')
        elif helper == 3:
            helper = 0
            action = 0
        else:
            helper += 1
            action = 0
            
        #print('action: ', action)

        screen =  np.array(ImageGrab.grab(bbox=(0,110,550,830)))
        new_screen, observation, reward, done, flag, num_of_space, total_time, stop = process_img(screen, action, flag, num_of_space, total_time, time_increment, prev_obs, stop)

        if len(prev_obs)>0 and flag == 1:
                game_memory.append([prev_obs, action])
                print('observation: ', observation)
                print('score: ', scores)
                print('totaltime: ', total_time)
        
        if(not done):
            scores += reward
            
        time_increment = time.time()-last_time
        print('Loop took {} seconds'.format(time_increment))
        last_time = time.time()

        cv2.imshow('window2', new_screen)
        #print('observation: ', observation)
        #print('done: ', done)
        #print('flag: ', flag)
        #print('score: ', scores)
        #print('totaltime: ', total_time)
        #print('stop: ', stop)
        #cv2.imshow('window',cv2.cvtColor(screen,cv2.COLOR_BGR2RGB))
        prev_obs = observation

        if(done):
            
            games += 1
            print('score: ', scores)
            
            if scores >= score_req:
                accepeted_scores.append(scores)
                for data in game_memory:
                    if data[1] == 1:
                        output = [0,1]
                    elif data[1] == 0:
                        output = [1,0]
                    training_data.append([data[0], output])
                    
            scores = 0
            game_memory = []
            pre_obs = np.array((0,0,0,0,0,0,0))
            total_time = 0
            num_of_space = 0
            pagui.press('space')

            print('game NO.', games)
            print('------------------------------------------')
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break  

    training_data_save = numpy.array(training_data)
    numpy.save('project_training_data_2.py', training_data_save)

    print('Average accepeted score:', mean(accepeted_scores))
    print('Median accepeted score:', median(accepeted_scores))
    print(Counter(accepeted_scores))
    #print('data: ', training_data)

    return training_data



def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = numpy.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]),1)
    Y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0])) 

    model.fit({'input':X},{'targets':Y}, n_epoch=2, snapshot_step=500, show_metric=True,
              run_id='project on flappy bird')

    return model


def play_with_model(num_of_games, model):

    #    
    training_data = []
    scores = []
    accepeted_scores = []

    #
    last_time = time.time()
    action = 1
    flag = 0
    time_increment = 0.0
    total_time = 0
    num_of_space = 0
    scores = 0
    prev_obs = []
    stop = 0

    #
    games = 0
    helper = 0
    #
    game_memory = []
    
    while (games < num_of_games):


        if flag == 1:
            print(model.predict(prev_obs.reshape(-1,len(prev_obs),1)))
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            print('prev_obs:', prev_obs)
            if action == 1:
                pagui.press('space')
        elif helper == 0:
            helper += 1
            action = 1
            pagui.press('space')
        elif helper == 3:
            helper = 0
            action = 0
        else:
            helper += 1
            action = 0


        print('action: ', action)

        screen =  np.array(ImageGrab.grab(bbox=(0,110,550,830)))
        new_screen, observation, reward, done, flag, num_of_space, total_time, stop = process_img(screen, action, flag, num_of_space, total_time, time_increment, prev_obs, stop)

        if len(prev_obs)>0:
                game_memory.append([prev_obs, action])
        
        if(not done):
            scores += reward
            
        time_increment = time.time()-last_time
        print('Loop took {} seconds'.format(time_increment))
        last_time = time.time()

        cv2.imshow('window2', new_screen)
        #print('observation: ', observation)
        #print('done: ', done)
        #print('flag: ', flag)
        #print('score: ', scores)
        #print('totaltime: ', total_time)
        #print('stop: ', stop)
        #cv2.imshow('window',cv2.cvtColor(screen,cv2.COLOR_BGR2RGB))
        prev_obs = observation

        if(done):
            
            games += 1
            print('score: ', scores)
            
            if scores >= score_req:
                accepeted_scores.append(scores)
                for data in game_memory:
                    if data[1] == 1:
                        output = [0,1]
                    elif data[1] == 0:
                        output = [1,0]
                    training_data.append([data[0], output])
                    
            scores = 0
            game_memory = []
            pre_obs = []
            total_time = 0
            num_of_space = 0
            pagui.press('space')

            print('game NO.', games)
            print('------------------------------------------')
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break  

    training_data_save = numpy.array(training_data)
    numpy.save('new_project_training_data_2.py', training_data_save)

    print('Average accepeted score:', mean(accepeted_scores))
    print('Median accepeted score:', median(accepeted_scores))
    print(Counter(accepeted_scores))
    #print('data: ', training_data)

    return training_data






#give time to change to game window
for j in range(0,3):
    print('starting in: ', 3-j, ' second')
    time.sleep(1)

training_data = initial_population()
#print(training_data)
#training_data = np.load('project_training_data_1.py.npy')
#print(training_data)
model = train_model(training_data)
play_with_model(20, model)

print('DONE!')


