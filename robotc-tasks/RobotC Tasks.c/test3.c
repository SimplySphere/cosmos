
task main()
{

setMotorSpeed(motorB, 50);
setMotorSpeed(motorC, 20);

sleep(4860);

setMotorSpeed(motorB, 0);
setMotorSpeed(motorC, 0);

sleep(100);

}
