
task main()
{

repeatUntil(getTouchValue(S1) == 1)
 {
   setMotorSpeed(motorB, 50);
   setMotorSpeed(motorC, 50);
 }

 setMotorSpeed(motorB, 0);
 setMotorSpeed(motorC, 0);

}
