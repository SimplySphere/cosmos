
task main()
{

int i

for (i = 0; i < 4; i++) {
	setMotorSpeed(motorB, 50);
	setMotorSpeed(motorC, 50);

	sleep(800);

	setMotorSpeed(motorB, -50);
	setMotorSpeed(motorC, 50);

	sleep(397);

	setMotorSpeed(motorB, 0);
	setMotorSpeed(motorC, 0);

	sleep(200);
}

}
