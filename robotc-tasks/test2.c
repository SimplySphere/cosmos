
task main()
{

setMotorSpeed(motorB, -50);
setMotorSpeed(motorC, 50);

sleep(1410);

setMotorSpeed(motorB, 0);
setMotorSpeed(motorC, 0);

sleep(100);

}
