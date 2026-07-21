
task main()
{

setMotorSpeed(motorB, 40);
setMotorSpeed(motorC, 20);

sleep(7300);

setMotorSpeed(motorB, 20);
setMotorSpeed(motorC, 40);

sleep(7300);

setMotorSpeed(motorB, 0);
setMotorSpeed(motorC, 0);

sleep(100);

}
