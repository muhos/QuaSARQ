
#include "input.hpp"
#include "control.hpp"

namespace QuaSARQ {

    void printUsage(int argc, char** argv, bool verbose)
    {
        LOGHEADER(0, 3, "Build");
        getCPUInfo();
        getBuildInfo();
        getGPUInfo();
        LOGHEADER(0, 5, "Usage");
        LOG0("");
        LOG1(" %spulsar%s [<circuit>.<qasm or stim>][<option> ...]", CSIM, CNORMAL);
        LOG0("");
        AvailOptions& avail_opts = ARG::opts();
        std::sort(avail_opts.data(), avail_opts.end(), ARG::ARG_CMP());
        arg_t prev_type = nullptr;
        LOG0("");
        LOG0(" Options:");
        for (int i = 0; i < avail_opts.size(); i++) {
            if (avail_opts[i]->type != prev_type) LOG0("\n");
            avail_opts[i]->help(verbose);
            prev_type = avail_opts[i]->type;
        }
        LOG0("");
        LOG1("  %s-h or --help  print available options.%s", CHELP, CNORMAL);
        LOG1("  %s--helpmore    print available options with verbose message.%s", CHELP, CNORMAL);
        LOG0("");
        LOGRULER('-', RULELEN);
        exit(EXIT_SUCCESS);
    }

    int parseArguments(int& argc, char** argv)
    {
        if (argc <= 1) return false;
        int dashes = (argv[1][0] == '-') + (argv[1][1] == '-');
        if ((dashes & 1) && argv[1][1] == 'h')
            printUsage(argc, argv);
        else if ((dashes & 2) && hasstr(argv[1], "help")) {
            if (hasstr(argv[1], "more"))
                printUsage(argc, argv, true);
            else
                printUsage(argc, argv);
        }
        struct stat st;
        int ispath = canAccess(argv[1], st);
        ispath += canAccess(argv[2], st);
        AvailOptions& avail_opts = ARG::opts();
        for (int i = 1 + ispath; i < argc; i++) {
            const size_t arglen = strlen(argv[i]);
            if (arglen == 1)
                LOGERROR("unknown input \"%s\". Use '-h or --help' for help.", argv[i]);
            else if (arglen > 1) {
                const char* arg = argv[i];
                int dashes = (arg[0] == '-') + (arg[1] == '-');
                if (!dashes)
                    LOGERROR("unknown input \"%s\". Use '-h or --help' for help.", argv[i]);
                else if ((dashes & 1) && arg[1] == 'h')
                    printUsage(argc, argv);
                if ((dashes & 2) && hasstr(arg, "help")) {
                    if (hasstr(arg, "more"))
                        printUsage(argc, argv, true);
                    else
                        printUsage(argc, argv);
                }
                else {
                    int k = 0;
                    bool parsed = false;
                    while (k < avail_opts.size() && !(parsed = avail_opts[k++]->parse(argv[i])));
                    if (!parsed)  LOGERROR("unknown input \"%s\". Use '-h or --help' for help.", argv[i]);
                }
            }
        }
        return ispath;
    }

    void printArguments(const int& argc) {
        const AvailOptions& avail_opts = ARG::opts();
        if (argc > 1) {
            LOGHEADER(1, 3, "Options");
            int i = 0, j = 0;
            const int MAX_PER_LINE = 4;
            int last_parsed = 0;
            for (i = 0; i < avail_opts.size(); i++)
                if (avail_opts[i]->isParsed())
                    last_parsed = i;
            for (i = 0, j = 0; i < avail_opts.size(); i++) {
                if (avail_opts[i]->isParsed()) {
                    if (j++ % MAX_PER_LINE == 0) PUTCH(' ');
                    avail_opts[i]->printArgument();
                    if (j % MAX_PER_LINE == 0) { PUTCH('\n'); }
                    else if (i == last_parsed) { PUTCH('\n'); }
                }
            }
        }
    }

}