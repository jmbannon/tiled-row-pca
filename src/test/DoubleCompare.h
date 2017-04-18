#include <stdbool.h>

#ifndef __DOUBLE_COMPARE_H__
#define __DOUBLE_COMPARE_H__

#define DEFAULT_EPSILON 1e-3

bool DoubleCompare_epsilon(double a, double b, double epsilon);
bool DoubleCompare(double a, double b);

#endif