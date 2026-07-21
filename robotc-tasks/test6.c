
task main()
{

int i

for (i = 0; i < 5; i++) {
	setMotorSpeed(motorB, 40);
	setMotorSpeed(motorC, 40);

	sleep(1800);

	setMotorSpeed(motorB, -50);
	setMotorSpeed(motorC, 50);

	sleep(615);

	setMotorSpeed(motorB, 0);
	setMotorSpeed(motorC, 0);

	sleep(200);
}

}
