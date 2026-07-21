
task main()
{

setMotorSpeed(motorB, 40);
setMotorSpeed(motorC, 40);

sleep(1800);

setMotorSpeed(motorB, -40);
setMotorSpeed(motorC, -40);

sleep(900);

setMotorSpeed(motorB, -50);
setMotorSpeed(motorC, 50);

sleep(385);

setMotorSpeed(motorB, 40);
setMotorSpeed(motorC, 40);

sleep(700);

setMotorSpeed(motorB, -50);
setMotorSpeed(motorC, 50);

sleep(385);

setMotorSpeed(motorB, 40);
setMotorSpeed(motorC, 40);

sleep(900);

setMotorSpeed(motorB, -40);
setMotorSpeed(motorC, -40);

sleep(1800);

}
