-----------------------------------------------------------------------
Rock Raiders 2018/19

Package: Jimbo_Kinematics

Author: Connor McGowan

Date: 10/29/18

Description: A package for calculating the forward and inverse kinematics
			 for the robotic arm mounted on the rover for the 2018-19
			 University Rover Challenge
			 

-----------------------------------------------------------------------

OK, here's the stuff that you actually care about.

The main functions you're gonna use are in the Forward_Kin and
Inverse_Kin modules.

These are broken up into pairs, one of each for every end effector.

The Forward_Kin functions take the joint parameters as inputs and tell
you the position and orientation of the arm.

The Inverse_Kin functions take a desired position and orientation and
give you the joint parameters you need to get there.



You might also want to include the Actuator_Conversions module.

It lets you convert between angle values and distances for the linear
actuator joints.

The angles are more useful to analyze the positioning of the arm, but
you need the distances when you're actuating those two joints.



The coordinate axes are defined as they are in the CAD for the chasis.

From the rover's perspective:
-Positive X is left
-Positive Y is up
-Positive Z is forward



Arm Definitions

P_mn is the vector (in the world frame) from joint m to joint n.

R_mn is the SO(3) rotation matrix that converts orientations
from the m frame to the n frame.

Zero configuration: The position is the arm when all inputs are 0.
-This is arbitrary, but has been chosen to simplify calculations

Zero Configuration for Jimbo
-Facing directly forward
-First link straight up
-Second link at right angle to first link, pointing straight out
-All end effectors pointing in same direction as second link



The arm parameters are defined in the Kinematic_Utils module.

At the bottom of the file, you can define the distances for the linear 
actuators that are considered to be the arm's zero position.

All arm dimensions are also defined at the bottom of the file.

Angle limits for the second and third joint are set in the 
convert_to_distance function in Actuator_Conversions.py


Convention Stuff

All distances are in units of inches.
All angles are in units of radians.

Forward_Kin functions ouput a transformation matrix, containing
the position and orientation.

Inverse_Kin functions output a matrix of nx1 vectors, each defining
a set of valid joint parameters.
-If any number in a solution is NaN, that solution is invalid.