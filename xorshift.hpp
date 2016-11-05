// http://kitayuta.hatenablog.com/entry/2014/11/15/230145

#include <limits>
#include <random>

class xor128{
public:
    using uint = std::uint32_t;
    using result_type = uint;
    static constexpr uint min(){ return std::numeric_limits<uint>::min(); }
    static constexpr uint max(){ return std::numeric_limits<uint>::max(); }
    uint operator()(){ return random(); }
    xor128() {
        std::random_device rd;
        w=rd();
        for(int i=0;i<10;i++){  // 数回空読み(不要?)
            random();
        }
    }
    explicit xor128(uint s){ w=s; }  // 与えられたシードで初期化
private:
    uint x=123456789u,y=362436069u,z=521288629u,w;
    uint random() {
        uint t = x ^ (x << 11);
        x = y; y = z; z = w;
        return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }
};
