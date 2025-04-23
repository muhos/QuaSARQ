#pragma once

#include "version.hpp"
#include "definitions.hpp"

#if defined(_WIN32)
#pragma execution_character_set("utf-8")
#endif

void LOGFANCYBANNER(const char* VER);

#define CBANNER CREPORT