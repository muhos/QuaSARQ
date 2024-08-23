#ifndef __BANNER_
#define __BANNER_

#include "version.hpp"
#include "definitions.hpp"

#if defined(_WIN32)
#pragma execution_character_set("utf-8")
#endif

void LOGFANCYBANNER(const char* VER);

#define CBANNER CREPORT


#endif