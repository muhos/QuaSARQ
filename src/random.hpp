#pragma once

#include <cassert>
#include "datatypes.hpp"

namespace QuaSARQ {

    /**
     * Random number generator based on Marsaglia's
     * shr3 (shift rotate) and Kiss (Keep it simple
     * stupid) constant. Randomness is up to 2^32.
    */
    class Random {
        uint32 _seed_;

    public:
                      Random          () : _seed_(1) {}
                      Random          (const uint32& seed) : _seed_(seed) { assert(_seed_); }
        inline void   init            (const uint32& seed) { _seed_ = seed; assert(_seed_); }
        inline uint32 seed            () const { return _seed_; }
        inline void   seed            (const uint32& new_seed) { _seed_ = new_seed; }
        inline uint32 irand           () {
            _seed_ ^= _seed_ << 13;
            _seed_ ^= _seed_ >> 17;
            _seed_ ^= _seed_ << 5;
            return _seed_;
        }
        inline double drand           () {
            return irand() * 2.328306e-10;
        }
        inline bool   brand           () {
            const double fraction = drand();
            assert(fraction >= 0 && fraction < 1);
            return uint32(2 * fraction);
        }
    };

}