//
// Created by richard on 9/24/24.
//

#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#ifdef WIN32
#pragma warning(disable : 4996)
#endif

// includes, project
#include "exception.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// includes, timer, string parsing, image helpers
#include "helper_image.h" // helper functions for image compare, dump, data comparisons
#include "helper_string.h"// helper functions for string parsing
#include "helper_timer.h" // helper functions for timers

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#endif//HELPER_FUNCTIONS_H
