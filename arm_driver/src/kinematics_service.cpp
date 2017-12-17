
/**
#File Name kinematics_service.cpp
#Authors David Michelman (miched@rpi.edu)
This node creates two services, /get_arm_ik and /get_arm_fk. They compute forward and inverse kinematics respectively.
This node can be launched with the following command: $ rosrun arm_driver kinematics_service
Forward kinematics can be tested like so: $ rosservice call get_arm_fk '[0.25, 0.0, 0.0, 0.0, 0.0, 0.0]'
Inverse kinematics can be tested like so: $ rosservice call get_arm_ik 'target_xyz: [0.25, 0.0, 0.0] 
target_angles: [1.0, -1.0, 0.5]'  (the new line is required)

Forward kinematics will always be sucessful unless an incorrect number of arm joints is passed in. Inverse kinematics
might not find a solution, in which case the "success" field in the service response will be false.

This node does not subscribe to any any topics or use any services. It does depend on include/ikfast/generated_cpp/ik.cpp
and include/ikfast/openrave-0.9.0/python/ikfast.h though.
*/


#define IKFAST_NO_MAIN  // turns off the main function in the generated ik.cpp file, won't compile otherwise

#include <cmath>
#include <ros/ros.h>
#include "arm_driver/ik_service.h"
#include "arm_driver/fk_service.h"
#include "ikfast/generated_cpp/ik.cpp"
#include "ikfast/openrave-0.9.0/python/ikfast.h"
#include <stdio.h>

// callback for get_arm_ik service
bool getIK(arm_driver::ik_service::Request  &req, arm_driver::ik_service::Response &res) {
  // turn the ik_service message into IK-Fast inputs
  IkSolutionList<IkReal> solutions;
  std::vector<IkReal> vfree(GetNumFreeParameters());
  IkReal eerot[9],eetrans[3];
  if (req.target_angles.size() == 9) {
    eerot[0] = req.target_angles[0]; eerot[1] = req.target_angles[1]; eerot[2] = req.target_angles[2]; eetrans[0] = req.target_xyz[0];
    eerot[3] = req.target_angles[3]; eerot[4] = req.target_angles[4]; eerot[5] = req.target_angles[5]; eetrans[1] = req.target_xyz[1];
    eerot[6] = req.target_angles[6]; eerot[7] = req.target_angles[7]; eerot[8] = req.target_angles[8]; eetrans[2] = req.target_xyz[2];
  }
  else if (req.target_angles.size() == 3) {
     // turn the yaw, pitch, and roll into a rotation matrix. 
     // Code taken from http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
    double ch = cos(req.target_angles[0]);
    double sh = sin(req.target_angles[0]);
    double ca = cos(req.target_angles[1]);
    double sa = sin(req.target_angles[1]);
    double cb = cos(req.target_angles[2]);
    double sb = sin(req.target_angles[2]);

    eerot[0] = ch * ca;
    eerot[1] = sh*sb - ch*sa*cb;
    eerot[2] = ch*sa*sb + sh*cb;
    eerot[3] = sa;
    eerot[4] = ca*cb;
    eerot[5] = -ca*sb;
    eerot[6] = -sh*ca;
    eerot[7] = sh*sa*cb + ch*sb;
    eerot[8] = -sh*sa*sb + ch*cb;
  }
  else  // imporoper arguments were passed in.
    return false;

  // actually compute the inverse kinematics
  bool success = ComputeIk(eetrans, eerot, vfree.size() > 0 ? &vfree[0] : NULL, solutions);
  res.success = success;

  if(success && solutions.GetNumSolutions() > 0) {
    // at least one solution found. This code was coppied from a generated ikfast cpp file
    std::vector<IkReal> solvalues(GetNumJoints());
    const IkSolutionBase<IkReal>& sol = solutions.GetSolution(0);
    std::vector<IkReal> vsolfree(sol.GetFree().size());
    sol.GetSolution(&solvalues[0],vsolfree.size()>0?&vsolfree[0]:NULL);
    for( std::size_t j = 0; j < solvalues.size(); ++j)
      res.angles.push_back(solvalues[j]);
  }
  return true;
}


// callback for get_arm_fk service
bool getFK(arm_driver::fk_service::Request  &req, arm_driver::fk_service::Response &res) {

  IkReal jointAngles[GetNumJoints()];
  IkReal eerot[9],eetrans[3];
  if (req.joint_angles.size() == GetNumJoints()) {
    for(int i = 0; i < GetNumJoints(); i++)
      jointAngles[i] = req.joint_angles.at(i);
    ComputeFk(jointAngles, eetrans, eerot);
    for(int i = 0; i < 3; i++)
      res.xyz.push_back(eetrans[i]);
    for(int i = 0; i < 9; i++)
      res.rotation_matrix.push_back(eerot[i]);
    return true;
  }
  else  // improper arguments were passed in
    return false;
}


int main(int argc, char **argv)
{
  // std::cout << "kinematics service started" << std::endl;
  ros::init(argc, argv, "kinematics_service");
  ros::NodeHandle n;
  ros::ServiceServer ikService = n.advertiseService("get_arm_ik", getIK);
  ros::ServiceServer fkService = n.advertiseService("get_arm_fk", getFK);
  ROS_INFO("kinematics server initialized.");
  ros::spin();
  return 0;
}