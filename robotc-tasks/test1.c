
task main()
{

setMotorSpeed(motorB, 100);
setMotorSpeed(motorC, 100);

sleep(900);

setMotorSpeed(motorB, 0);
setMotorSpeed(motorC, 0);

sleep(1000);

}
