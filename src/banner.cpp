#include "banner.hpp"

void LOGFANCYBANNER(const char* VER) 
{
#if defined(_WIN32)
    SetConsoleOutputCP(65001);
#endif

    const charu8_t* title1 = u8"    ██████                          █████████    █████████   ███████████      ██████   ";
    const charu8_t* title2 = u8"  ███░░░░███                       ███░░░░░███  ███░░░░░███ ░░███░░░░░███   ███░░░░███ ";
    const charu8_t* title3 = u8" ███    ░░███ █████ ████  ██████  ░███    ░░░  ░███    ░███  ░███    ░███  ███    ░░███";
    const charu8_t* title4 = u8"░███     ░███░░███ ░███  ░░░░░███ ░░█████████  ░███████████  ░██████████  ░███     ░███";
    const charu8_t* title5 = u8"░███   ██░███ ░███ ░███   ███████  ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███ ░███   ██░███";
    const charu8_t* title6 = u8"░░███ ░░████  ░███ ░███  ███░░███  ███    ░███ ░███    ░███  ░███    ░███ ░░███ ░░████ ";
    const charu8_t* title7 = u8" ░░░██████░██ ░░████████░░████████░░█████████  █████   █████ █████   █████ ░░░██████░██";
    const charu8_t* title8 = u8"   ░░░░░░ ░░   ░░░░░░░░  ░░░░░░░░  ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░    ░░░░░░ ░░ ";
    const charu8_t* name_version = u8"   Copyright\u00A9 Muhammad Osama Mahmoud                                            ";
    size_t len = 85 + 1;
    if (RULERLEN < (len - 3)) LOGERROR("ruler length is smaller than the title (%zd)", len);
    size_t gap = (RULERLEN - len - 3) / 2;
    
    PUTCH('\n');
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title1, CNORMAL);  
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title2, CNORMAL); 
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title3, CNORMAL);  
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title4, CNORMAL);  
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title5, CNORMAL); 
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title6, CNORMAL); 
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title7, CNORMAL);
    REPCH(' ', gap);
    PRINT("%s%s%s\n", CBANNER, title8, CNORMAL);
    REPCH(' ', gap);
    PRINT("%s%s%6s%s\n", CBANNER, name_version, VER, CNORMAL);
    PUTCH('\n');
}