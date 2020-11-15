import sim
import simConst
import simpleTest
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import math
from scipy.spatial.transform import Rotation as R

#plt.ion()




# Round down data to 2 digits
def round2(data_in):
    if isinstance(data_in, list):
        for i in range(len(data_in)):
            data_in[i] = round(data_in[i], 2)
        return data_in
    else:
        return round(data_in, 2)

# Calculate distance
def distance(pos_cur, pos_des):
    #pos_cur, pos_des are [x, y] format
    return round2( math.sqrt( (pos_des[1]-pos_cur[1])**2 + (pos_des[0]-pos_cur[0])**2  ) )

# Calculate angles
def theta(pos_cur, pos_des):
    #pos_cur, pos_des are [x, y] format
    angle = round2( math.atan2( pos_des[1]-pos_cur[1], pos_des[0]-pos_cur[0]) )
    # angle =  math.atan( (pos_des[1]-pos_cur[1]) / (pos_des[0]-pos_cur[0]) ) 
    '''
    if abs(angle) > 2.4:
        angle = 3.14- abs(angle)
    else: 
        angle = angle
    '''
    #return round2( math.atan2( abs(pos_des[1]-pos_cur[1]), abs(pos_des[0]-pos_cur[0])) )
    print('toT angle:', angle)
    return angle

# Input velocity and desired steering angle
# Output differential drive angular velocity of wheels
def control2diff(velocity, steer_rate, wheel_sepration, wheel_R):
    v_l = (2 * velocity - (wheel_sepration) * steer_rate) / 2 * wheel_R
    v_r = (2 * velocity + (wheel_sepration) * steer_rate) / 2 * wheel_R

    w_l = v_l / wheel_R
    w_r = v_r / wheel_R

    return w_l, w_r


if __name__ == '__main__':
    print('Program started')
    sim.simxFinish(-1)  # just in case, close all opened connections

    # Connect to CoppeliaSim, set a very large time-out for blocking commands
    clientID = sim.simxStart('127.0.0.1', 19997, True, True, -500000, 5)

    # Read Inward spiral path points
    f = open('/home/kai/RSN/VREP_test/mowerPath.txt', 'r')
    path_point = []
    for line in f.readlines():
        path_point.append( [round2(float(line.split()[0])), round2(float(line.split()[1]))] )
    print(path_point)

    if clientID != -1:
        print('Connected to remote API server')

        emptyBuff = bytearray()

        # Start the simulation:
        sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)

        # Connect to Kinect RGB and depth sensor
        _, cameraHandle = sim.simxGetObjectHandle(clientID, 'kinect_rgb', sim.simx_opmode_oneshot_wait)
        _, depthHandle = sim.simxGetObjectHandle(clientID, 'kinect_depth', sim.simx_opmode_oneshot_wait)
        _, res, img = sim.simxGetVisionSensorImage(clientID, cameraHandle, 0, operationMode=sim.simx_opmode_streaming)
        _, res, depth = sim.simxGetVisionSensorDepthBuffer(clientID, depthHandle, operationMode=sim.simx_opmode_streaming)
        _, agent_handle = sim.simxGetObjectHandle(clientID, 'Robotnik_Summit_XL', sim.simx_opmode_blocking)
        #_, cube = sim.simxGetObjectHandle(clientID, 'Cuboid', sim.simx_opmode_oneshot_wait) 
        #Connect to Robotnik_Summit_XL wheels
        _, robot_fl_wheel = sim.simxGetObjectHandle(clientID, 'joint_front_left_wheel', sim.simx_opmode_blocking)
        _, robot_fr_wheel = sim.simxGetObjectHandle(clientID, 'joint_front_right_wheel', sim.simx_opmode_blocking)
        _, robot_bl_wheel = sim.simxGetObjectHandle(clientID, 'joint_back_left_wheel', sim.simx_opmode_blocking)
        _, robot_br_wheel = sim.simxGetObjectHandle(clientID, 'joint_back_right_wheel', sim.simx_opmode_blocking)
        _, field_handle = sim.simxGetObjectHandle(clientID, 'ResizableFloor_5_25', sim.simx_opmode_oneshot_wait)
        time.sleep(0.1)
        
        wheel_seperation = 0.235 
        wheel_radius = 0.117 
        
        try:
            
            for i in range(300):
                # Read the RGB image and depth from Kinect sensor
                _, res, img = sim.simxGetVisionSensorImage(clientID, cameraHandle, 0, operationMode=sim.simx_opmode_buffer)
                _, res, depth = sim.simxGetVisionSensorDepthBuffer(clientID, depthHandle, operationMode=sim.simx_opmode_buffer)
                img = np.array(img,dtype=np.uint8)
                img.resize((res[1],res[0],3))
                img = np.flip(img, axis=0)
                #print('depth:', depth)
                #print('img is', img)
                #plt.imshow(img)
                #plt.show()
                time.sleep(0.5)
                
                # Read the velocity
                _, linear_velocity, angular_v = sim.simxGetObjectVelocity(clientID, robot_br_wheel, sim.simx_opmode_streaming)
                linear_velocity = round2( linear_velocity[:-1])
                angular_v = round2( angular_v )
                agent_velocity = round2( math.sqrt( linear_velocity[0]**2 + linear_velocity[1]**2 ) )

                # Read current position
                _, agent_position = sim.simxGetObjectPosition(clientID, agent_handle, -1, sim.simx_opmode_streaming)
                _, agent_position = sim.simxGetObjectPosition(clientID, agent_handle, -1, sim.simx_opmode_buffer)
                agent_position = round2(agent_position[:-1])

                _, agent_ori = sim.simxGetObjectOrientation(clientID, agent_handle, -1, sim.simx_opmode_streaming)
                _, agent_ori = sim.simxGetObjectOrientation(clientID, agent_handle, -1, sim.simx_opmode_buffer)
                agent_ori = round2(agent_ori)
                rotation = R.from_euler('xyz', agent_ori)
                
                des_distance = distance(agent_position,path_point[1])
                
                if des_distance < 1:
                    path_point.pop(0)
                    print('poped')

                # Use the wheel differential spedd to control the speed
                v_desire = 1.0
                #Theta = abs( theta(agent_position,path_point[1]) )

                agent_angle = agent_ori[2]
                '''
                if agent_angle > 1.57:
                    agent_angle = agent_angle - 1.57
                else:
                    agent_angle = agent_angle
                '''
                the_desire = 17* (  theta(agent_position,path_point[1]) - agent_angle )

                l_v, r_v = control2diff(v_desire, the_desire, wheel_seperation, wheel_radius)
                #l_v = 20
                #r_v = -20
                print( 'ori', agent_ori[2], 'theta', the_desire/10, 'Position:', agent_position,  
                         'dis', des_distance)
                #, 'ori:', rotation.as_matrix()'ag v:', angular_v,'theta of T', the_desire + agent_ori[2],'at V:', agent_velocity,

                # Right
                err_code = sim.simxSetJointTargetVelocity(clientID, robot_br_wheel, -r_v, sim.simx_opmode_streaming)
                err_code = sim.simxSetJointTargetVelocity(clientID, robot_fr_wheel, -r_v, sim.simx_opmode_streaming)
                # Left
                err_code = sim.simxSetJointTargetVelocity(clientID, robot_bl_wheel, l_v, sim.simx_opmode_streaming)
                err_code = sim.simxSetJointTargetVelocity(clientID, robot_fl_wheel, l_v, sim.simx_opmode_streaming)

                
            
            #if res==sim.simx_return_ok:
            #    res=sim.simxSetVisionSensorImage(clientID,cameraHandle,image,0,sim.simx_opmode_oneshot)
            
            print(res)
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass

        # Stop simulation:
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)

        # Now close the connection to CoppeliaSim:
        sim.simxFinish(clientID)