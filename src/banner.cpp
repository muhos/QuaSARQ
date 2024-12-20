#include "banner.hpp"

void LOGFANCYBANNER(const char* VER) 
{
#if defined(_WIN32)
    SetConsoleOutputCP(65001);
#endif

    const char* title1 = u8"    ██████                          █████████    █████████   ███████████      ██████   ";
    const char* title2 = u8"  ███░░░░███                       ███░░░░░███  ███░░░░░███ ░░███░░░░░███   ███░░░░███ ";
    const char* title3 = u8" ███    ░░███ █████ ████  ██████  ░███    ░░░  ░███    ░███  ░███    ░███  ███    ░░███";
    const char* title4 = u8"░███     ░███░░███ ░███  ░░░░░███ ░░█████████  ░███████████  ░██████████  ░███     ░███";
    const char* title5 = u8"░███   ██░███ ░███ ░███   ███████  ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███ ░███   ██░███";
    const char* title6 = u8"░░███ ░░████  ░███ ░███  ███░░███  ███    ░███ ░███    ░███  ░███    ░███ ░░███ ░░████ ";
    const char* title7 = u8" ░░░██████░██ ░░████████░░████████░░█████████  █████   █████ █████   █████ ░░░██████░██";
    const char* title8 = u8"   ░░░░░░ ░░   ░░░░░░░░  ░░░░░░░░  ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░    ░░░░░░ ░░ ";
    const char* name_version = u8"   Copyright\u00A9 Muhammad Osama Mahmoud                                            ";
    size_t len = 85 + 1;
    if (RULERLEN < (len - 3)) LOGERROR("ruler length is smaller than the title (%zd)", len);
    size_t gap = (RULERLEN - len - 3) / 2;
    PRINT(PREFIX);
    PUTCH('\n');
    PRINT(PREFIX);
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title1, CNORMAL);
    PRINT(PREFIX);
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title2, CNORMAL);
    PRINT(PREFIX);
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title3, CNORMAL);
    PRINT(PREFIX);
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title4, CNORMAL);
    PRINT(PREFIX);
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title5, CNORMAL);
    PRINT(PREFIX);
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title6, CNORMAL);
    PRINT(PREFIX);
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title7, CNORMAL);
    PRINT(PREFIX);
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title8, CNORMAL);
    PRINT(PREFIX);
    REPCH(' ', gap);
    PRINT("%s%s%6s%s\n", CBANNER, name_version, VER, CNORMAL);
    PRINT(PREFIX);
    PUTCH('\n');
}