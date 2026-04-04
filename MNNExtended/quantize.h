#ifndef QUANTIZE_H
#define QUANTIZE_H
#include "MyModules.h"

namespace MyVectorQuantizer
{
using namespace DonNotKnowHowToNameIt;

class VectorQuantizer2 : public MyModule
{
public:
    std::shared_ptr<MyModule> embedding{nullptr};
    
    VectorQuantizer2(int n_e, int e_dim, bool enable_remap) :
        embedding(std::make_shared<Embedding>(n_e, e_dim))
    {
        register_module
        (
            "embedding",
            embedding
        );
        
        if (enable_remap)
        {
            
        }
    }
};

}//namespace MyVectorQuantizer

#endif