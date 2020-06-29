import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def render_traj(traj, vf=None, scaler=None):

    fig1 = plt.figure(1,figsize=plt.figaspect(0.5))
    fig1.clear()
    plt.figure(fig1.number)
    fig1.set_size_inches(8, 8, forward=True)
    gridspec.GridSpec(4,2)
    t = np.asarray(traj['t'])
    t1 = t[0:-1]

    pos = np.asarray(traj['position'])
    vel = np.asarray(traj['velocity'])
    norm_pos = np.linalg.norm(pos,axis=1)
    norm_vel = np.linalg.norm(vel,axis=1)

    plt.subplot2grid( (4,2) , (0,0) )
    plt.plot(t, pos[:,0],'r',label='X')
    plt.plot(t, pos[:,1],'b',label='Y')
    plt.plot(t, pos[:,2],'g',label='Z')
    plt.plot(t,norm_pos,'k',label='N')
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Position (m)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)


    plt.subplot2grid( (4,2) , (0,1) )
    pc = np.asarray(traj['seeker_angles'])[0:-1]
    plt.plot(t1,pc[:,0],'r',label='px 0' )
    plt.plot(t1,pc[:,1],'b',label='px 1' )
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Coords')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    plt.subplot2grid( (4,2) , (1,0) ) 
    opt_flow=np.asarray(traj['optical_flow'])[0:-1]
    plt.plot(t1,np.asarray(opt_flow[:,0]),'r',label='du')
    plt.plot(t1,np.asarray(opt_flow[:,1]),'b',label='dv')
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_xlabel("time (s)")
    plt.gca().set_ylabel('optical flow')
    plt.grid(True)

    plt.subplot2grid( (4,2) , (1,1) )
    tcv = np.asarray(traj['theta_cv'])
    plt.plot(t,tcv,'r',label='Thteta_CV')
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_xlabel("Time")
    plt.gca().set_ylabel('Theta CV')
    plt.grid(True)


    plt.subplot2grid( (4,2) , (2,0))
    plt.plot(t,vel[:,0],'r',label='X')
    plt.plot(t,vel[:,1],'b',label='Y')
    plt.plot(t,vel[:,2],'g',label='Z')
    plt.plot(t,norm_vel,'k',label='N')
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Velocity (m/s)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)


    thrust = np.asarray(traj['thrust'])
    plt.subplot2grid( (4,2) , (2,1) )
    plt.plot(t,thrust[:,0],'r',label='X')
    plt.plot(t,thrust[:,1],'b',label='Y')
    plt.plot(t,thrust[:,2],'g',label='Z')
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Thrust (N)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    attitude = np.asarray(traj['attitude_321'])
    plt.subplot2grid( (4,2) , (3,0) )
    colors = ['r','b','k','g']
    for i in range(attitude.shape[1]):
        plt.plot(t,attitude[:,i],colors[i],label='q' + '%d' % (i))
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_ylabel('Attitude (rad)')
    plt.gca().set_xlabel("Time (s)")
    plt.grid(True)

    w = np.asarray(traj['w'])
    plt.subplot2grid( (4,2) , (3,1) )
    plt.plot(t, w[:,0],'r',label='X')
    plt.plot(t, w[:,1],'b',label='Y')
    plt.plot(t, w[:,2],'g',label='Z')
    plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3,
            ncol=5, mode="expand", borderaxespad=0.)
    plt.gca().set_xlabel("Time (s)")
    plt.gca().set_ylabel('Rot. Velocity (rad/s)')
    plt.grid(True)


    plt.tight_layout(h_pad=3.0)
    fig1.canvas.draw()

