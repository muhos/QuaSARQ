#pragma once

#include "datatypes.hpp"
#include "gatetypes.hpp"

namespace QuaSARQ {

    // QC  = 1-qubit gate on control qubit
    // QT  = 1-qubit gate on target qubit
    // QCT = 2-qubit gate on (control, target)
    // QTC = 2-qubit gate on (target, control)
    enum   QSelector : byte_t { QC = 0, QT = 1, QCT = 2, QTC = 3 };
    struct PhaseOp        { Gatetypes type; QSelector sel; };
    struct CliffordDecomp { const char* name; PhaseOp ops[12]; int nops; bool is_2q; };

    static constexpr CliffordDecomp TABLE[] = {
        { "H_YZ",        {{H,QC},{S,QC},{H,QC},{S,QC},{S,QC}},                             5,  false },
        { "H_XY",        {{H,QC},{S,QC},{S,QC},{H,QC},{S,QC}},                             5,  false },
        { "XCX",         {{H,QC},{CX,QCT},{H,QC}},                                         3,  true  },
        { "XCY",         {{H,QC},{S_DAG,QT},{CX,QCT},{H,QC},{S,QT}},                       5,  true  },
        { "XCZ",         {{CX,QTC}},                                                       1,  true  },
        { "YCX",         {{S_DAG,QC},{H,QT},{CX,QTC},{S,QC},{H,QT}},                       5,  true  },
        { "YCY",         {{S_DAG,QC},{S_DAG,QT},{H,QC},{CX,QCT},{H,QC},{S,QC},{S,QT}},     7,  true  },
        { "YCZ",         {{S_DAG,QC},{CX,QTC},{S,QC}},                                     3,  true  },
        { "SQRT_XX",     {{H,QC},{CX,QCT},{H,QT},{S,QC},{S,QT},{H,QC},{H,QT}},             7,  true  },
        { "SQRT_XX_DAG", {{H,QC},{CX,QCT},{H,QT},{S_DAG,QC},{S_DAG,QT},{H,QC},{H,QT}},     7,  true  },
        { "SQRT_YY",     {{S_DAG,QC},{S_DAG,QT},{H,QC},{CX,QCT},{H,QT},
                            {S,QC},{S,QT},{H,QC},{H,QT},{S,QC},{S,QT}},                    11,  true  },
        { "SQRT_YY_DAG", {{S_DAG,QC},{S,QT},{H,QC},{CX,QCT},{H,QT},
                            {S,QC},{S,QT},{H,QC},{H,QT},{S,QC},{S_DAG,QT}},                11,  true  },
        { "SQRT_ZZ",     {{H,QT},{CX,QCT},{H,QT},{S,QC},{S,QT}},                           5,  true  },
        { "SQRT_ZZ_DAG", {{H,QT},{CX,QCT},{H,QT},{S_DAG,QC},{S_DAG,QT}},                   5,  true  },
        { "CXSWAP",      {{CX,QTC},{CX,QCT}},                                              2,  true  },
        { "CZSWAP",      {{H,QC},{CX,QCT},{CX,QTC},{H,QT}},                                4,  true  },
        { "SWAPCZ",      {{H,QC},{CX,QCT},{CX,QTC},{H,QT}},                                4,  true  },
        { "SWAPCX",      {{CX,QCT},{CX,QTC}},                                              2,  true  },
    };

}