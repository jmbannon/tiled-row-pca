#include "DoubleCompare.h"
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

bool DoubleCompare_epsilon(double a, double b, double epsilon) {
	//printf("%lf to %lf : %lf, %d\n", a, b, fabs(a - b), fabs(a-b) < epsilon);
	return fabs(a - b) < epsilon;
}

bool DoubleCompare(double a, double b) {
	DoubleCompare_epsilon(a, b, DEFAULT_EPSILON);
}